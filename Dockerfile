FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV TERM=xterm-256color
WORKDIR /tmp
ENV AXELERA_RUNTIME_DIR="/opt/axelera/runtime-1.4.0-1"
ENV AXELERA_DEVICE_DIR="/opt/axelera/device-1.4.0-1/omega"
ENV AXELERA_RISCV_TOOLCHAIN_DIR="/opt/axelera/riscv-gnu-newlib-toolchain-409b951ba662-7"
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
ENV AXELERA_FRAMEWORK="/data/voyager-sdk"
ENV AIPU_FIRMWARE_OMEGA="/opt/axelera/device-1.4.0-1/omega/bin/start_axelera_runtime.elf"
ENV AIPU_RUNTIME_STAGE0_OMEGA="/opt/axelera/device-1.4.0-1/omega/bin/start_axelera_runtime_stage0.bin"
ENV TVM_HOME="/opt/axelera/runtime-1.4.0-1/tvm/tvm-src"
ENV GST_PLUGIN_PATH="/opt/axelera/runtime-1.4.0-1/lib/gstreamer-1.0:/data/voyager-sdk/operators/lib"
ENV PYTHONPATH="/data/voyager-sdk:/opt/axelera/runtime-1.4.0-1/tvm/tvm-src"
ENV LD_LIBRARY_PATH="/opt/axelera/runtime-1.4.0-1/lib:/data/voyager-sdk/operators/lib"
ENV PATH="/opt/axelera/runtime-1.4.0-1/bin:/opt/axelera/riscv-gnu-newlib-toolchain-409b951ba662-7/bin:/home/aetina/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/go/bin"
ENV PKG_CONFIG_PATH="/opt/axelera/runtime-1.4.0-1/lib/pkgconfig:/data/voyager-sdk/operators/lib/pkgconfig"
RUN useradd --badname -m -u 1001 aetina
RUN groupadd -g 107 render
RUN groupadd -g 1004 axelera
RUN groupadd -g 106 kvm
RUN usermod -a -G sudo,video,render,kvm,axelera aetina
RUN echo 'aetina:aetina' | chpasswd
RUN mkdir -p /home/aetina/.local && chown -R aetina:aetina /home/aetina/.local
VOLUME /home/aetina/.local
