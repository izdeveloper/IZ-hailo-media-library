# Hailo-15 LPR / Webserver Project

This repository contains the Hailo Media Library / Hailo Analytics sources used to build and run the License Plate Recognition (LPR) pipeline on the Hailo-15 platform. The code is split between:

- the standalone LPR application in `IZ-hailo-media-library/hailo-analytics/apps/license_plate_recognition`
- the HTTP webserver pipeline in `IZ-hailo-media-library/hailo-analytics/apps/webserver`
- Hailo post-process shared libraries in `IZ-hailo-media-library/hailo-postprocess`

The project is source-based and is meant to be compiled for the target ARM board using a Poky/Yocto cross toolchain and the Hailo Dataflow Compiler toolchain for model compilation. 
And be plased on flexwatch camera whit hailo15h 

## 1. What the LPR app does

The standalone LPR application is implemented in [IZ-hailo-media-library/hailo-analytics/apps/license_plate_recognition/main.cpp](IZ-hailo-media-library/hailo-analytics/apps/license_plate_recognition/main.cpp). It creates a detection + OCR pipeline that includes:

- detection and tiling
- vehicle attribute inference
- plate state classification
- OCR recognition for the plate text
- LPR event analytics

The app accepts a Media Library config path, optional file source, host IP, UDP/ZMQ output parameters, and a tracking mode (`slow`, `fast`, or `balanced`). The default Media Library config is:

```bash
/etc/imaging/cfg/medialib_configs/ai_example_medialib_config.json
```

The core LPR pipeline is built in [IZ-hailo-media-library/hailo-analytics/apps/license_plate_recognition/lpr_pipeline_builder.cpp](IZ-hailo-media-library/hailo-analytics/apps/license_plate_recognition/lpr_pipeline_builder.cpp). It creates:

- `build_tiling_pipeline()`: keeps vehicle and license-plate detections and adds a lightweight tracker
- `build_vehicle_attributes_pipeline()`: runs `vehicle_attributes.hef` and loads `libvehicle_attributes_post.so`
- `build_classification_pipeline()`: runs `plate_state.hef` and loads `libstate_cls_post.so`
- `build_ocr_pipeline()`: runs `paddle_ocr_v5_mobile_recognition.hef` and loads `libocr_post.so`

The LPR analytics API is defined in [IZ-hailo-media-library/hailo-analytics/hailo_analytics_api/include/hailo_analytics/analytics/license_plate_recognition.hpp](IZ-hailo-media-library/hailo-analytics/hailo_analytics_api/include/hailo_analytics/analytics/license_plate_recognition.hpp). It shows that the OCR model is expected at:

```bash
/home/root/apps/license_plate_recognition/resources/paddle_ocr_v5_mobile_recognition.hef
```

and the corresponding post-process libraries are expected in `/usr/lib/hailo-post-processes/`.

## 2. LPR section in the webserver

The webserver variant is in [IZ-hailo-media-library/hailo-analytics/apps/webserver/main.cpp](IZ-hailo-media-library/hailo-analytics/apps/webserver/main.cpp) and [IZ-hailo-media-library/hailo-analytics/apps/webserver/pipeline/lpr_pipeline.cpp](IZ-hailo-media-library/hailo-analytics/apps/webserver/pipeline/lpr_pipeline.cpp).

The server selects the `LPR` pipeline when the webserver starts, then builds the same multi-stage LPR flow as the standalone app:

- `tiling_pipeline` (the defaule tiling pipeline from hailo)
- `vehicle_attributes_pipeline`
- `classification_pipeline`
- `ocr_pipeline`
- `lpr_event_engine_post`
- `lpr_event_sink`

The webserver exposes the following LPR API endpoints:

- `/api/v1/saved-events` — saved event JSON history
- `/api/v1/lpr-events` — live polling data for recent events
- `/api/v1/event-config` — event database size settings

The event sink does not just print a result. It:

1. finds the INEX classification object (`inex_event`)
2. encodes the current frame to JPEG
3. attaches the image to the event JSON
4. saves the event to `/home/root/apps/webserver/resources/configs/events_db.json`
5. uploads the event to the configured INEX HTTP endpoint

A key code path is the event upload logic in [IZ-hailo-media-library/hailo-analytics/apps/webserver/pipeline/lpr_pipeline.cpp](IZ-hailo-media-library/hailo-analytics/apps/webserver/pipeline/lpr_pipeline.cpp), where it reads the target endpoint from:

```bash
/home/root/apps/license_plate_recognition/resources/event_config_ip.json
```

and posts JSON data to the external endpoint configured there.

This is the real LPR “webserver” section of the project: it is not a front-end only feature; it is the live event ingestion and persistence layer for plate detections and OCR results.

## 3. Relevant project layout

The repo contains the Hailo libraries and application sources that this project depends on:

- `IZ-hailo-media-library/hailo-analytics/` — analytics apps and pipeline logic
- `IZ-hailo-media-library/hailo-postprocess/` — OCR / state / vehicle attribute / LPR event post-process shared libs

## 4. Install the Hailo Dataflow Compiler

### this section is more for the [INEX LPR MODEL ZOO](https://github.com/izdeveloper/INEX_LPR_Hailo.git) repo 

The PDF [IZ-hailo-media-library/M1707-00-Hailo-15_IP_Camera_AI_SDK_v1.0_Eng.pdf](IZ-hailo-media-library/M1707-00-Hailo-15_IP_Camera_AI_SDK_v1.0_Eng.pdf) is the authoritative Hailo-15 SDK document for this project. In the SDK installation section, the exact compiler version is:

- [`hailo_dataflow_compiler-5.1.0-py3-none-linux_x86_64.whl`](https://drive.google.com/file/d/1hI1MwMM1GYgkjzYpKAfc9V5lqSHbfe62/view?usp=drive_link)
 

```bash
pip install --upgrade pip
pip install hailo_dataflow_compiler-5.1.0-py3-none-linux_x86_64.whl
```

### Lock DFC dependencies

```bash
pip install setuptools==69.5.1 \
  tensorflow==2.18.0 tensorboard==2.18.0 keras==3.5.0 \
  numpy==1.26.4 protobuf==3.20.3 flatbuffers==24.3.25 \
  gast==0.4.0 typing-extensions==4.12.2 \
  onnx==1.16.0 onnxruntime==1.18.0 networkx==2.8.8
```

Test with:

```bash
hailo --help
```

## 5. Install SDK and Build Natively APPS

To build the app and the post prosses files you will need the cross compiler 

- Download the vision Processor [Hailo-15H Vision Processor Software Package SBC 2.X](https://drive.google.com/file/d/1J3dQBqVkiQda0-6AkG6vS6ZE-ew_voY2/view?usp=drive_link) 


Typical install flow:

```bash
sudo sh /home/manager/Desktop/prebuilt/sbc/sdk/poky-glibc-x86_64-core-imageminimal-armv8a-hailo15-sbc-toolchain-4.0.23.sh -y -d /opt/poky/4.0.23
```

Then clone the repo and follow this command order 

Build Environment Setup and Compilation

Loading SDK Environment Variables
Open linux/ubunto terminal
You must run it once per terminal session.
```bash
. /opt/poky/4.0.23/environment-setup-armv8a-poky-linux
```
Enter to the hailo-analytics to build the apps ot into hailo-postprocess for the post prosses files
Meson Build Configuration
Create a build directory including the Hailo-15H platform configuration.
```bash
meson build
```
Start Ninja Compilation
Generates the actual library and application binary.
```bash
ninja -C build
```

Then you will see the compiled files inside the build dirs 

### Importent node: For every model there is config json file that you can change but make sure this fit the model labels, In the lpr_event_sink json you can config the ip that the events will be send to this ip addresss 