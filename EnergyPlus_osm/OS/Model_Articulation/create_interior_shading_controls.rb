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

subSurfaceNames = ["GroundFloor_E_Window",
               "GroundFloor_S_Window 1",
               "GroundFloor_S_Window 2",
               "GroundFloor_W_Window",
               "mid_E_Window",
               "mid_S_Window",
               "mid_W_Window",
               "top_E_Window",
               "top_S_Window",
               "top_W_Window"]

subSurfacesByZoneHashForPrint = {}
subSurfacesByZoneHash = {}

subSurfaceNames.each do |n|
  s = m.getSubSurfaceByName(n).get
  space = s.space.get
  z = space.thermalZone.get
  if !subSurfacesByZoneHashForPrint.key?(z.nameString)
    subSurfacesByZoneHashForPrint[z.nameString] = []
    subSurfacesByZoneHash[z] = []
  end
  subSurfacesByZoneHashForPrint[z.nameString] << n
  subSurfacesByZoneHash[z] << s
end

nori = m.getShadingControls.size

shading_mat = m.getShadingMaterialByName('Int_Isotiss_Thermal Curtain_Only').get

avail_sch = m.getScheduleRulesetByName('INT_Curtain_Test').get

subSurfacesByZoneHash.each do |z, ss_list|
  sc = ShadingControl.new(shading_mat)
  # NightTime: if OAT is < 15°C, we apply the thermal curtain
  sc.setName("#{z.nameString} Interior Shading Control")
  sc.setShadingType('InteriorShade')
  sc.setSetpoint(15.0)
  sc.setShadingControlType('OnNightIfLowOutdoorTempAndOffDay')
  sc.setMultipleSurfaceControlType("Group")
  sc.setSchedule(avail_sch)
  ss_list.each{|ss| sc.addSubSurface(ss) }
end

puts "Start with #{nori} shading controls, ended with #{m.getShadingControls.size} for #{subSurfacesByZoneHash.size} Thermal Zones"
puts subSurfacesByZoneHashForPrint


m.save(path_save, true)

