
! Create a top-level team of 2 threads:
!$omp parallel nom_threads(2)
! Verify that we have (at least) two threads!

if (omp_thread_num == 0) then

   ! Branch that creates all the worker threads that will
   ! end up executing task functions...
  ! Create a thread team:
!$omp parallel






!$omp end parallel


else

   ! Branch for one or several threads that run the iterator


  ! Create a thread team:
!$omp parallel if(Want_parallel_iterator)




!$omp end parallel

end if
