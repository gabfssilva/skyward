from skyward.actors.messages import (
    NodeAvailable,
    NodeUnavailable,
    RegisterPressureObserver,
    SubmitBroadcast,
    SubmitTask,
    TaskResult,
    TaskSubmitted,
)

type TaskManagerMsg = (
    NodeAvailable | NodeUnavailable | TaskResult
    | SubmitTask | SubmitBroadcast | TaskSubmitted
    | RegisterPressureObserver
)
