# Run via `openstudio create_shading_controls.rb`

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

nori = m.getShadingControls.size

m.getShadingControls.each{|sc| sc.remove}

puts "Removed #{nori} Shading Controls"

m.save(path_save, true)
