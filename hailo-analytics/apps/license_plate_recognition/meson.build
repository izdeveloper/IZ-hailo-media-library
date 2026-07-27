################################################
# License Plate Recognition Application
################################################

lpr_install_dir = apps_install_dir + '/license_plate_recognition'

# LPR Pipeline Builder library (reusable dependency)
lpr_pipeline_builder_lib = static_library(
  'lpr_pipeline_builder',
  sources: ['lpr_pipeline_builder.cpp'],
  cpp_args: hailo_lib_args,
  dependencies: libhailo_analytics_dep,
  gnu_symbol_visibility: 'default',
)

lpr_pipeline_builder_dep = declare_dependency(
  link_with: lpr_pipeline_builder_lib,
  include_directories: include_directories('.'),
  dependencies: libhailo_analytics_dep,
)

# LPR App
lpr_app_src = ['main.cpp']

# 다른 프로젝트(hailo-postprocess)의 빌드 산출물을 상대 위치로 참조
cpp = meson.get_compiler('cpp')

# hailo-analytics 소스 루트: .../hailo-media-library-1.11.0/hailo-analytics
# 형제 프로젝트 hailo-postprocess로 이동 (한 단계 위 → hailo-postprocess)
postprocess_root = meson.project_source_root() / '..' / 'hailo-postprocess'

oem_process_lib = cpp.find_library(
    'oem_process',
    dirs : [postprocess_root / 'build' / 'postprocess_tools' / 'src' / 'oem_process'],
    required : true,
)

oem_process_inc = include_directories(
    '..' / '..' / '..' / 'hailo-postprocess' / 'postprocess_tools' / 'src' / 'oem_process',
    is_system : true,
)

oem_process_dep = declare_dependency(
    dependencies : oem_process_lib,
    include_directories : oem_process_inc,
)

lpr_deps = [libhailo_analytics_dep, lpr_pipeline_builder_dep]

#lpr_deps += [oem_stage_dep]
#lpr_deps += [oem_process_dep]

executable('lpr_app',
  lpr_app_src,
  cpp_args : hailo_lib_args,
  # dependencies : [libhailo_analytics_dep, lpr_pipeline_builder_dep],
  dependencies : lpr_deps,
  gnu_symbol_visibility : 'default',
  install: true,
  install_dir: lpr_install_dir,
)
