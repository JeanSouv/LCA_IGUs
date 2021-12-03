# Run via `openstudio add_output_variables.rb`

require 'openstudio'

include OpenStudio::Model


path_load = File.expand_path("../Office_BXL.osm",File.dirname(__FILE__))
# path_save = File.expand_path("../model_out.osm",File.dirname(__FILE__))
path_save = path_load

# Helper to load a model in one line
def osload(path)
  translator = OpenStudio::OSVersion::VersionTranslator.new
  ospath = OpenStudio::Path.new(path)
  model = translator.loadModel(ospath)
  if model.empty?
      raise "Path '#{path}' is not a valid path to an OpenStudio Model"
  else
      model = model.get
  end
  return model
end

m = osload(path_load)

vars = [
  'Surface Window Shading Device Absorbed Solar Radiation Rate',
  'Surface Shading Device Is On Time Fraction',
  'Surface Window Transmitted Solar Radiation Rate',
  'Surface Storm Window On Off Status',
  'Surface Window Blind Slat Angle',
  'Surface Outside Face Solar Radiation Heat Gain Rate',
  'Surface Window Heat Gain Rate',
  'Surface Window Heat Loss Rate',
  'Surface Outside Face Incident Solar Radiation Rate per Area',
  'Surface Outside Face Incident Beam Solar Radiation Rate per Area',
  'Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area'
]

keyval = nil
# Limit to one south window
keyval = 'mid_S_Window'

freq = 'Timestep'

vars.each do |var|
  out_var = OutputVariable.new(var, m)
  out_var.setReportingFrequency('Timestep')
  if keyval
    out_var.setKeyValue(keyval)
  end
end

zone_vars = [
  'Zone Windows Total Transmitted Solar Radiation Rate',
  'Zone Windows Total Heat Gain Rate',
  'Zone Windows Total Heat Loss Rate',
  'Zone Mean Air Temperature',
]
keyval = 'mid_Offices_S_TZ'

zone_vars.each do |var|
  out_var = OutputVariable.new(var, m)
  out_var.setReportingFrequency('Timestep')
  if keyval
    out_var.setKeyValue(keyval)
  end
end

env_vars = [
  'Site Diffuse Solar Radiation Rate per Area',
  'Site Direct Solar Radiation Rate per Area',
  'Site Outdoor Air Drybulb Temperature',
]

env_vars.each do |var|
  out_var = OutputVariable.new(var, m)
  out_var.setReportingFrequency('Timestep')
end

m.save(path_save, true)
