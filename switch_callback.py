import elastica as ea
import numpy as np

class SwitchCallBack(ea.CallBackBaseClass):
    """
    Call back function for switch
    """

    def __init__(self, step_skip: int, callback_params: dict) -> None:
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(
        self, system: ea.typing.RodType, time: np.float64, current_step: int
    ) -> None:

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["acceleration"].append(system.acceleration_collection.copy())
            self.callback_params["force"].append(system.external_forces.copy())

            return