# Sources
oem_process_sources = [
    'OemProcess.cpp',
]

oem_process_incdir = include_directories('.')

# shared_library → liboem_process.so.1.11.0
oem_process_lib = shared_library(
    'oem_process',
    oem_process_sources,
    cpp_args: hailo_lib_args,
    include_directories: [incdir, oem_process_incdir],
    version: meson.project_version(),
    soversion: '1',
    install: true,
    install_dir: get_option('libdir'),
)

oem_process_dep = declare_dependency(
    include_directories: [oem_process_incdir],
    link_with: oem_process_lib,
)

# fwencd(다른 프로젝트)가 소비할 수 있도록 헤더 설치 + pkg-config 노출
install_headers('OemProcess.h', subdir : 'hailo')

pkgc.generate(
    oem_process_lib,
    name : 'oem-process',
    description : 'Hailo OEM Process shared library',
    filebase : 'oem-process',
    subdirs : ['hailo'],
)