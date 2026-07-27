# Sources
oem_stage_sources = [
    'OemStage.cpp',
]

# Include dirs
oem_stage_incdir = include_directories('.')

# shared_library → liboem_stage.so.1.11.0 생성
oem_stage_lib = shared_library(
    'oem_stage',
    oem_stage_sources,
    cpp_args: hailo_lib_args,
    include_directories: [incdir, oem_stage_incdir],
    version: meson.project_version(),   # → .so.1.11.0
    soversion: '1',                     # → .so.1 symlink
    install: true,
    install_dir: get_option('libdir'),
)

oem_stage_dep = declare_dependency(
    include_directories: [oem_stage_incdir],
    link_with: oem_stage_lib,
)