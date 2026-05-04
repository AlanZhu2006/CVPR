# Source this before running HikRobot MVS Python/ROS2 tools.
export MVS_HOME="${MVS_HOME:-/opt/MVS}"
export MVCAM_COMMON_RUNENV="$MVS_HOME/lib"
export MVCAM_SDK_PATH="$MVS_HOME"
export MVCAM_GENICAM_CLPROTOCOL="$MVS_HOME/lib/CLProtocol"
export ALLUSERSPROFILE="$MVS_HOME/MVFG"
export LD_LIBRARY_PATH="$MVS_HOME/lib/aarch64:$MVS_HOME/lib/64:$MVS_HOME/bin:$LD_LIBRARY_PATH"
export PYTHONPATH="$MVS_HOME/Samples/aarch64/Python/MvImport:$MVS_HOME/Samples/arm64/Python/MvImport:$MVS_HOME/Samples/64/Python/MvImport:$PYTHONPATH"
