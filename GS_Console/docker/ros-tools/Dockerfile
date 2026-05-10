FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-rosbridge-suite \
    ros-noetic-tf2-tools \
    ros-noetic-rqt-tf-tree \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
