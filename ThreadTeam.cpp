/*
 * \file    ThreadTeam.cpp
 */

#include "ThreadTeam.h"

#include <cstdio>
#include <iostream>
#include <stdexcept>

/**
 * Instantiate a thread team that, at any point in time, can have no more than
 * nMaxThreads threads in existence.  Thread teams must contain at least one
 * thread.
 * 
 * This routine initializes the state of all possible threads to THREAD_IDLE
 * and the team will be in the state TEAM_IDLE.
 *
 * \param  nMaxThreads The maximum permissible number of threads in the team.
 *                     Zero threads is considered to be a logical error.
 * \param  name        The name of the thread team for debug use.
 */
ThreadTeam::ThreadTeam(const unsigned int nMaxThreads,
                       const std::string name)
    : nMaxThreads_(nMaxThreads),
      name_(name),
      isTerminating_(false),
      queue_(),
      state_(TEAM_IDLE),
      threadStates_(nullptr),
      taskFcn_(nullptr),
      threads_(nullptr),
      threadData_(nullptr),
      threadReceiver_(nullptr),
      workReceiver_(nullptr)
{
    if (nMaxThreads_ == 0) {
        throw std::logic_error("[ThreadTeam::ThreadTeam] "
                               "Empty thread teams disallowed");
    }

    pthread_mutex_init(&teamMutex_, NULL);

    // Use of the mutex should not be necessary since no other code
    // that could use the object's members can run until construction
    // begins.  However, I use it to be defensive and under the assumption
    // that teams won't be instantiated often.
    pthread_mutex_lock(&teamMutex_);

    // TODO: Do we need to set more attributes?
    // TODO: Are the detached threads being handles appropriately so that
    //       we don't have any resource loss?
    pthread_attr_init(&attr_);
    pthread_attr_setdetachstate(&attr_, PTHREAD_CREATE_DETACHED);

    pthread_cond_init(&threadIdling_, NULL);
    pthread_cond_init(&activateThread_, NULL);
    pthread_cond_init(&checkQueue_, NULL);
    pthread_cond_init(&threadTerminated_, NULL);

    int rc = 0;
    threadStates_ = new threadState[nMaxThreads_];
    threads_      = new   pthread_t[nMaxThreads_];
    threadData_   = new  ThreadData[nMaxThreads_];
    for (unsigned int i=0; i<nMaxThreads_; ++i) {
        threadData_[i].tId = i;
        threadData_[i].team = this;
        rc = pthread_create(&threads_[i], &attr_, *threadRoutine,
                            reinterpret_cast<void*>(&(threadData_[i])));
        if (rc != 0) {
            pthread_mutex_unlock(&teamMutex_);
            throw std::runtime_error("[ThreadTeam::ThreadTeam] "
                                     "Unable to create thread");
        }
        threadStates_[i] = STARTING;
    }

    // Wait until all threads have started running their routine
    do {
        pthread_cond_wait(&threadIdling_, &teamMutex_);
    } while (nIdleThreads() < nMaxThreads_);
    state_ = TEAM_IDLE;

    pthread_mutex_unlock(&teamMutex_);

    printf("%d threads created in %s ThreadTeam\n", nMaxThreads_, name_.c_str());
    printf("%s ThreadTeam is idle\n", name_.c_str());
}

/**
 * Destroy the thread team.  It is considered a logical or runtime error if any
 * of the following is true at the time of destruction:
 *   - there is a thread in the team that is *not* idle
 *   - the team still has work pending
 */
ThreadTeam::~ThreadTeam(void) {
    pthread_mutex_lock(&teamMutex_);

    isTerminating_ = true;

    // TODO: How to handle errors discovered in destructor?

    // Destruction should only happen if the team is idle,
    // which means that all threads are idle.
    if (state_ != TEAM_IDLE) {
        std::cerr << "[ThreadTeam::~ThreadTeam] "
                     "ERROR - Thread team is still running\n\n";
    } else if (nIdleThreads() != nMaxThreads_) {
        std::cerr << "[ThreadTeam::~ThreadTeam] "
                     "ERROR - All threads must be idle\n\n";
    }

    // Sanity check that there is no work in the queue
    if (!queue_.empty()) {
        std::cerr << "[ThreadTeam::~ThreadTeam] "
                     "ERROR - Work still pending\n\n";
    }

    // All threads are sleeping and should receive this signal
    pthread_cond_broadcast(&activateThread_);

    // Block until all threads have terminated
    unsigned int nTerminated = 0;
    do {
        pthread_cond_wait(&threadTerminated_, &teamMutex_);

        nTerminated = 0;
        for (unsigned int i=0; i<nMaxThreads_; ++i) {
            if (threadStates_[i] == TERMINATING) {
                ++nTerminated;
            }
        }
    } while (nTerminated < nMaxThreads_);

    delete [] threadData_;
    delete [] threads_;
    delete [] threadStates_;

    pthread_cond_destroy(&threadTerminated_);
    pthread_cond_destroy(&checkQueue_);
    pthread_cond_destroy(&activateThread_);
    pthread_cond_destroy(&threadIdling_);

    pthread_attr_destroy(&attr_);

    pthread_mutex_unlock(&teamMutex_);

    pthread_mutex_destroy(&teamMutex_);

    printf("%d threads terminated in %s ThreadTeam\n", nMaxThreads_, name_.c_str());
    printf("%s ThreadTeam has terminated\n", name_.c_str());
}

/**
 * 
 *
 * \return 
 */
std::string  ThreadTeam::name(void) const {
    return name_;
}

/**
 * Obtain the total number of threads that the thread team may contain at any
 * point in time.
 *
 * \return The number of threads.
 */
unsigned int ThreadTeam::nMaximumThreads(void) const {
    return nMaxThreads_;
}

/**
 * Obtain the total number of threads that are idling at the current time.
 *
 * \warning Code that calls this routine must first use teamMutex_ to
 *          successfully obtain a lock on team thread resources.
 *
 * \return The number of threads.
 */
unsigned int ThreadTeam::nIdleThreads(void) const {
    int nIdle = 0;
    for (unsigned int i=0; i<nMaxThreads_; ++i) {
        if (threadStates_[i] == THREAD_IDLE) {
            ++nIdle; 
        }
    }

    return nIdle;
}

/**
 * Activate nThreads more threads in the team.
 *
 * A thread subscriber team could also be a thread publisher team.  If such a
 * team finishes its work before its publisher does, then this method will
 * immediately push new threads along to its subscriber.
 *
 * It is considered to be a logical error if activating this many threads would
 * exceed the maximum number of threads allowed in the team.
 * 
 * As the startTask() method sets the original number of threads to use to
 * execute a task with the team, calling this method with the team in the idle
 * state is considered to be a logical error.
 *
 * \param   nThreads   The number of new threads to activate.
 */
void ThreadTeam::increaseThreadCount(const unsigned int nThreads) {
    if (nThreads == 0) {
        throw std::runtime_error("[ThreadTeam::increaseThreadCount] "
                                 "Zero thread increase given");
    }
    
    pthread_mutex_lock(&teamMutex_);

    if (threadReceiver_) {
        if (     state_ == TEAM_IDLE
            || ((state_ == RUNNING_CLOSED_QUEUE) && (queue_.empty()))) {
            threadReceiver_->increaseThreadCount(nThreads);
            pthread_mutex_unlock(&teamMutex_);
            return;
        }
    } else if (state_ == TEAM_IDLE) {
        // Client code should call the publisher wait() before the subscriber
        // wait().  Therefore, the publisher would not be sending threads
        // to its idle subscriber.
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error("[ThreadTeam::increaseThreadCount] "
                                 "Idle team with no thread receiver does not "
                                 "need more threads");
    } else if (nThreads > nIdleThreads()) {
        // If there are no idle threads, then the following signal would not be
        // caught and we would lose a thread resource.
        //
        // Maximum thread counts in each team should be properly setup so that
        // this is impossible.
        pthread_mutex_unlock(&teamMutex_);
        throw std::runtime_error("[ThreadTeam::increaseThreadCount] "
                                 "Number of threads in team is too large");
    }

    pthread_mutex_unlock(&teamMutex_);

    for (unsigned int i=0; i<nThreads; ++i) {
        // Wake the threads individually so that they can immediately
        // start their work.  The calling thread is likely going to idle next,
        // so we can slow it down.
        pthread_mutex_lock(&teamMutex_);
        pthread_cond_signal(&activateThread_);
        pthread_mutex_unlock(&teamMutex_);
    }
}

/**
 * Initiate an task execution cycle that will use a thread team that will
 * initially contain nThreads threads.  The number of threads active in the team
 * can subsequently be increased with the increaseThreadCount() method.
 *
 * The thread team must be in the idle state when this method is called.  This
 * can occur if
 *   1) startTask() has not yet been called for the team or
 *   2) the wait() method was the last method called and has terminated.
 *
 * After calling this method, the team can still accept work via the enqueue()
 * method.
 *
 * Note that passing zero threads is a valid input.  This might be useful if
 * this team will later be pushed threads from a thread publisher.
 *
 * \param  taskFcn   A function that is applied to each unit of work.  When this
 *                   function terminates, it is considered that the task has
 *                   been executed on that unit and it is removed from the
 *                   queue.
 * \param  nThreads  The number of threads to use at the start of this execution
 *                   cycle.
 */
void ThreadTeam::startTask(TASK_FCN* taskFcn, const unsigned int nThreads) {
    if (!taskFcn) {
        throw std::logic_error("[ThreadTeam::startTask] "
                               "Null task function pointer");
    }

    pthread_mutex_lock(&teamMutex_);

    taskFcn_ = taskFcn;

    if (nThreads > nMaxThreads_) {
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error("[ThreadTeam::startTask] "
                               "Too many starting threads");
    } else if (state_ != TEAM_IDLE) {
        pthread_mutex_unlock(&teamMutex_);
        std::logic_error("[ThreadTeam::startTask] "
                         "Execution cycle already underway");
    } else if (!queue_.empty()) {
        pthread_mutex_unlock(&teamMutex_);
        std::logic_error("[ThreadTeam::startTask] "
                         "Queue must be empty before starting thread team");
    } else if (nIdleThreads() < nMaxThreads_) {
        pthread_mutex_unlock(&teamMutex_);
        std::logic_error("[ThreadTeam::startTask] "
                         "Threads already active");
    }

    for (unsigned int i=0; i<nThreads; ++i) {
        pthread_cond_signal(&activateThread_);
    }

    state_ = RUNNING_OPEN_QUEUE;
    printf("%s ThreadTeam is running and accepting work\n", name_.c_str());

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Add a unit of work to the thread team's workload.  This method can only
 * be invoked when
 *   1) the thread team is idle or
 *   2) after calling startTask() but before calling closeTask().
 *
 * \param   work  The unit of work to add.
 */
void ThreadTeam::enqueue(const int work) {
    pthread_mutex_lock(&teamMutex_);

    if (state_ == RUNNING_CLOSED_QUEUE) {
        pthread_mutex_unlock(&teamMutex_);
        std::logic_error("[ThreadTeam::enqueue] "
                         "Cannot queue after wait() called");
    }

    queue_.push(work);
    printf("%s work %d enqueued\n", name_.c_str(), work);

    // Signal a single waiting thread to look for work
    pthread_cond_signal(&checkQueue_);

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * A non-blocking routine that is called after startTask() but before wait() to
 * inform the thread team that client code has finished adding to the queue all
 *  work to be evaluated by the team during the current execution cycle.
 *
 * \param
 * \return
 */
void ThreadTeam::closeTask(void) {
    // Ideally, the call to wait() would communicate to the team that no more
    // work will be added.  However, in the case that there are more than one
    // thread teams simultaneously running, the blocking nature of wait() would
    // lead to potential inefficiencies.  In particular, consider the case where
    // we call wait() for team A before calling wait() for team B.  If team B
    // finishes its work before A, then all team B threads will sit idle because
    // team B does not yet know that no more work will be added.
    pthread_mutex_lock(&teamMutex_);

    printf("%s ThreadTeam is running and has all work\n", name_.c_str());
    state_ = RUNNING_CLOSED_QUEUE;

    pthread_mutex_unlock(&teamMutex_);
}

/**
 * Client code calls this method to block its execution until all work units
 * have been processed and all threads in the team have transitioned to idle.
 * When this method returns, the thread team will therefore be idle.
 *
 * This routine can only be called after calling closeTask().
 *
 * If a thread team is a work or thread subscriber, then the subscriber team's
 * wait() function must be called after the wait() function of its publisher
 * teams.
 */
void ThreadTeam::wait(void) {
    // TODO: Review this in light of teams with no threads and no work
    // TODO: If we have a team that pushes work or threads to another
    //       team, how do we make certain that this doesn't finish 
    //       if the receiver team is still spinning up?
    // TODO: Enumerate these based on transitions?  When are states changed,
    // when is data added?  Are we discovering these enumerations along the way?
    // Possible entry scenarios
    //   1) No work in the queue and no more work will be added
    //      a) all threads idle
    //         i)   All active threads were computing when closeTask() was
    //              called
    //              - set team to idle
    //         ii)  startTask() given zero threads and team not setup as thread
    //              subscriber
    //              - resources wasted
    //              - could be detected if we store nStartThreads and check if
    //                team is a thread subscriber
    //              - TODO: Worth the effort?  Allow client code to setup
    //                      bad pipelines?
    //              - set team to idle
    //         iii) bug in thread team implementation
    //              - shouldn't happen
    //      b) threads waiting and possibly some idle threads
    //         i)   team had active threads but some or all threads were in wait
    //              already when closeTask() was called
    //              - set team to idle
    //      c) threads computing and possibly some idle threads
    //      d) threads computing and waiting and possibly some idle threads
    //   2) Work in queue but no more work will be added
    //      a) all threads idle
    //      b) threads waiting and possibly some idle threads
    //      c) threads computing and possibly some idle threads
    //      d) threads computing and waiting and possibly some idle threads
    //
    // At this point, we could have
    //   1) no more active threads 
    //      Example - only one thread in team
    //   2) all active threads waiting
    //      Example - N>1 threads created and only one work unit
    //   3) all active threads computing
    //      Example - more work units than threads and all work
    //                units in queue before startTask() called
    //   4) a mixture of computing and waiting threads
    //      Example - more threads than work units in queue and more
    //                than one work unit in queue when startTask() called
    //
    // A thread can only terminate if the work queue is empty and this
    // method has been called.  Therefore, there is no more work in the
    // queue for waiting threads and we signal *all* waiting threads to check
    // the queue so that they can determine that they can terminate.
    //
    // No computing threads can check the queue yet since we hold the mutex.
    // Therefore, they will determine when they get the mutex that they can 
    // terminate.  Hence, they will not wait and it is unimportant that they
    // miss this signal.

    pthread_mutex_lock(&teamMutex_);

    if (state_ != RUNNING_CLOSED_QUEUE) {
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error("[ThreadTeam::wait] "
                               "closeTask() not called yet");
    } else if (!queue_.empty() && (nIdleThreads() == nMaxThreads_)) {
        // Work enqueued, but no threads made active yet.
        //
        // This could occur if
        //   1) the team code is buggy and allows for threads to go idle before
        //      all work is enqueued (shouldn't happen) or
        //   2) startTask() is given zero threads and the team is not setup
        //      as a thread subscriber.
        pthread_mutex_unlock(&teamMutex_);
        throw std::logic_error("[ThreadTeam::wait] "
                               "closeTask() not called yet");
    }

    if (queue_.empty()) {
        if (nIdleThreads() == nMaxThreads_) {
            // Thread team never received work and no threads created.
            // This resets team status so that it can start queueing work again.
            state_ = TEAM_IDLE;
            printf("%s ThreadTeam is idle\n", name_.c_str());

            pthread_mutex_unlock(&teamMutex_);
            return;
        } else {
            // No more work and some threads possibly waiting.
            //  => Inform all waiting threads to check the queue and go idle
            //
            // Upon terminating their work, all computing threads should
            // discover that there is no more work and should go idle.
            pthread_cond_broadcast(&checkQueue_);
        }
    }

    // Block until all threads in team have transitioned to idle
    while (nIdleThreads() < nMaxThreads_) {
        pthread_cond_wait(&threadIdling_, &teamMutex_);
    }
  
    // Sanity check the team's state
    if (!queue_.empty()) {
        pthread_mutex_unlock(&teamMutex_);
        std::logic_error("[ThreadTeam::wait] "
                         "No more threads to carry out pending work"); 
    }

    // This resets team status so that it can start queueing work again.
    state_ = TEAM_IDLE;
    printf("%s ThreadTeam is idle\n", name_.c_str());

    pthread_mutex_unlock(&teamMutex_);
}

/**
 *
 *
 * \param
 * \return
 */
void* ThreadTeam::threadRoutine(void* varg) {
    pthread_detach(pthread_self());

    ThreadData* data = reinterpret_cast<ThreadData*>(varg);
    if (!data) {
        pthread_exit(NULL);
        throw std::runtime_error("[ThreadTeam::threadRoutine] "
                                 "Null thread data pointer");
    }

    unsigned int tId = data->tId;
    ThreadTeam* team = data->team;
    if (!team) {
        pthread_exit(NULL);
        throw std::runtime_error("[ThreadTeam::threadRoutine] "
                                 "Null thread team pointer");
    }

    // All threads must initially wait and indicate that they have started their
    // routine.
    pthread_mutex_lock(&(team->teamMutex_));

    team->threadStates_[tId] = THREAD_IDLE;
    pthread_cond_signal(&(team->threadIdling_));

    // Wait for first call to startTask()
    pthread_cond_wait(&(team->activateThread_), &(team->teamMutex_));
    printf("[%s %d] Activated\n", team->name_.c_str(), tId);

    pthread_mutex_unlock(&(team->teamMutex_));

    int         work = 0;
    bool        isEmpty = false;
    teamState   state = TEAM_IDLE;
    bool        foundWork = false;
    while (true) {
        foundWork = false;

        pthread_mutex_lock(&(team->teamMutex_));
        isEmpty = team->queue_.empty();
        state = team->state_;

        if ((team->isTerminating_) && (state == TEAM_IDLE)) {
            team->threadStates_[tId] = TERMINATING;
            pthread_cond_signal(&(team->threadTerminated_));

            pthread_mutex_unlock(&(team->teamMutex_));

            pthread_exit(NULL);
            return nullptr;
        } else if ((team->isTerminating_) && (state != TEAM_IDLE)) {
            pthread_mutex_unlock(&(team->teamMutex_));
            throw std::runtime_error("[ThreadPool::threadRoutine] "
                                     "Thread cannot terminate if team is not idle");
        } else if (isEmpty && (state == RUNNING_CLOSED_QUEUE)) {
            // No more work in queue and no more work can be added since a
            // thread has called wait
            //    => thread can now terminate.
            // Update state before signaling that thread is terminating.
            team->threadStates_[tId] = THREAD_IDLE;

            // Inform wait() so that it can block until all threads are idling
            pthread_cond_signal(&(team->threadIdling_));

            printf("[%s %d] Thread transitioning to idle\n", team->name_.c_str(), tId);

            // Transfer thread resources to receiver thread team
            if (team->threadReceiver_) {
                team->threadReceiver_->increaseThreadCount(1);
            }

            pthread_cond_wait(&(team->activateThread_), &(team->teamMutex_));
            printf("[%s %d] Activated\n", team->name_.c_str(), tId);
        } else if (isEmpty && (state == RUNNING_OPEN_QUEUE)) {
            // There is still the potential for more work to be added.
            // Therefore, we don't want the thread to terminate, but rather
            // to wait for the signal that it should check the queue again.
            printf("[%s %d] Waiting for work\n", team->name_.c_str(), tId);
            team->threadStates_[tId] = WAITING;
            pthread_cond_wait(&(team->checkQueue_), &(team->teamMutex_)); 
            printf("[%s %d] Rechecking thread team state\n", team->name_.c_str(), tId);

            // State could have changed during wait, so recheck state
            // with explicit calls
            if (!team->queue_.empty()) {
                // work was enqueued
                work = team->queue_.front();
                team->queue_.pop();
                foundWork = true;

                // Since the state upon entry is no wait, we cannot
                // conclude that there will be no more work.
            }

            // It is possible for
            //       1) thread A to be in wait state
            //       2) work to be enqueued
            //       3) thread B to pop that work first
            //       4) thread A to receive the signal to check the queue
            //       5) thread A finds the queue empty
            //
            // It is also possible that a thread terminated and wait() informed
            // this thread to recheck the queue so that it can discover that there
            // is no more work and that it should therefore terminate as well.
            // 
            // In these cases, the thread will go back into a wait state or
            // terminate.
        } else if (  !isEmpty && 
                   ((state == RUNNING_OPEN_QUEUE) || (state == RUNNING_CLOSED_QUEUE))) {
            work = team->queue_.front();
            team->queue_.pop();
            foundWork = true;

            // Update to see if this thread left the queue empty.
            // If this is the last work to be done, signal to all waiting 
            // threads that they should check the queue and therefore
            // determine that they should terminate.
            if (team->queue_.empty() && (state == RUNNING_CLOSED_QUEUE)) {
                pthread_cond_broadcast(&(team->checkQueue_));
            }
        } else {
            // TODO: Print out state information to help understand why this
            //       occurred.
            team->printState(tId);
            pthread_mutex_unlock(&(team->teamMutex_));
            throw std::logic_error("[ThreadTeam::threadRoutine] "
                                   "Invalid thread control flow"); 
        }
        pthread_mutex_unlock(&(team->teamMutex_));

        // Release the mutex before executing computation on work
        if (foundWork) {
            team->threadStates_[tId] = COMPUTING;
            printf("[%s %d] Dequeued work %d\n", team->name_.c_str(), tId, work);
            team->taskFcn_(tId, team->name_, work);

            // Send work to next thread team in the pipeline if it exists
            if (team->workReceiver_) {
                team->workReceiver_->enqueue(work); 

                if (team->queue_.empty() && (state == RUNNING_CLOSED_QUEUE)) {
                    team->workReceiver_->closeTask();
                }
            }
        }
    }

    throw std::logic_error("[ThreadTeam::threadRoutine] "
                           "Inavlid thread control flow");
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeam::attachThreadReceiver(ThreadTeam* receiver) {
    pthread_mutex_lock(&teamMutex_);

    if (!receiver) {
        throw std::logic_error("[ThreadTeam::attachThreadReceiver] "
                               "Null receiver thread team given");
    } else if (receiver == this) {
        throw std::logic_error("[ThreadTeam::attachThreadReceiver] "
                               "Cannot attach the team to itself");
    } else if (threadReceiver_) {
        throw std::logic_error("[ThreadTeam::attachThreadReceiver] "
                               "A receiver team is already attached");
    }

    threadReceiver_ = receiver;

    pthread_mutex_unlock(&teamMutex_);
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeam::detachThreadReceiver(void) {
    pthread_mutex_lock(&teamMutex_);

    threadReceiver_ = nullptr;

    pthread_mutex_unlock(&teamMutex_);
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeam::attachWorkReceiver(ThreadTeam* receiver) {
    pthread_mutex_lock(&teamMutex_);

    if (!receiver) {
        throw std::logic_error("[ThreadTeam::attachWorkReceiver] "
                               "Null receiver thread team given");
    } else if (receiver == this) {
        throw std::logic_error("[ThreadTeam::attachWorkReceiver] "
                               "Cannot attach the team to itself");
    } else if (workReceiver_) {
        throw std::logic_error("[ThreadTeam::attachWorkReceiver] "
                               "A receiver team is already attached");
    }

    workReceiver_ = receiver;

    pthread_mutex_unlock(&teamMutex_);
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeam::detachWorkReceiver(void) {
    pthread_mutex_lock(&teamMutex_);

    workReceiver_ = nullptr;

    pthread_mutex_unlock(&teamMutex_);
}

/**
 *
 *
 * \param
 * \return
 */
void ThreadTeam::printState(const unsigned int tId) const {
    std::string teamState    = "";
    switch(state_) {
    case TEAM_IDLE:
        teamState = "Idle";
        break;
    case RUNNING_OPEN_QUEUE:
        teamState = "Executing Task/Queue Open";
        break;
    case RUNNING_CLOSED_QUEUE:
        teamState = "Executing Task/Queue Closed";
        break;
    default:
        throw std::logic_error("[ThreadTeam::printState] "
                               "Unknown team state");
    }

    printf("[%s %d] %s Thread Team State Snapshot\n",
           name_.c_str(), tId, name_.c_str());
    printf("[%s %d] -------------------------------------------------------\n",
           name_.c_str(), tId);
    printf("[%s %d] \tThreadTeam State\t\t%s\n",
           name_.c_str(), tId,
           teamState.c_str());
    printf("[%s %d] \tUnits of Work in Queue\t\t%d\n", 
           name_.c_str(), tId,
           queue_.size());
    printf("[%s %d] \tN Threads in Team\t\t%d\n",
           name_.c_str(), tId, nMaxThreads_);
    printf("[%s %d] \tThread States\n",
           name_.c_str(), tId);
    std::string threadState = "";
    for (unsigned int i=0; i<nMaxThreads_; ++i) {
        switch(threadStates_[i]) {
        case STARTING:
            threadState = "Starting";
            break;
        case THREAD_IDLE:
            threadState = "Idle";
            break;
        case COMPUTING:
            threadState = "Computing";
            break;
        case WAITING:
            threadState = "Waiting";
            break;
        case TERMINATING:
            threadState = "Terminating";
            break;
        default:
            throw std::logic_error("[ThreadTeam::printState] "
                                   "Unknown thread state");
        }
        printf("[%s %d] \t\tThread %d\t\t%s\n",
               name_.c_str(), tId,
               i, threadState.c_str());
    }

    if (threadReceiver_) {
        printf("[%s %d] \tThread Receiver\t\t\t%s\n",
               name_.c_str(), tId, threadReceiver_->name().c_str());
    } else {
        printf("[%s %d] \tNo Thread Receiver\n",
               name_.c_str(), tId);
    }

    if (workReceiver_) {
        printf("[%s %d] \tWork Receiver\t\t\t%s\n",
               name_.c_str(), tId, workReceiver_->name().c_str());
    } else {
        printf("[%s %d] \tNo Work Receiver\n",
               name_.c_str(), tId);
    }
}

