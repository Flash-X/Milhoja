!<
!!
!!  

module RuntimeTest_data
    implicit none

    !!!!! ORCHESTRATION SYSTEM OFFLINE TOOLCHAIN
    ! The variables created here shall be written into the code by the offline
    ! toolchain.  These are variables that would normally be passed to tasks
    ! executed by the runtime as parameters.  However, such variability in task
    ! interface cannot be allowed.  Therefore, before executing a task with the
    ! runtime, client code must first set the value of the task's variables
    ! declared here and the task must get those values using the same variables.
    integer, save :: task1_iteration = -1
    real,    save :: task1_dt        = -1.0
end module RuntimeTest_data

