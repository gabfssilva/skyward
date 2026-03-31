from skyward.actors.messages import (
    NodeAvailable,
    NodeUnavailable,
    RegisterPressureObserver,
    SubmitBroadcast,
    SubmitTask,
    TaskFailed,
    TaskInterrupted,
    TaskSubmitted,
    TaskSucceeded,
)

type TaskManagerMsg = (
    NodeAvailable | NodeUnavailable
    | TaskSucceeded | TaskFailed | TaskInterrupted
    | SubmitTask | SubmitBroadcast | TaskSubmitted
    | RegisterPressureObserver
)
