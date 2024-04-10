# trace generated using paraview version 5.11.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

for i in range(101,115):
    # create a new 'XML PolyData Reader'
    dP2vtp = XMLPolyDataReader(registrationName='DP' + str(i) + '.vtp', FileName=['U:\\dLight\\fwdFacingStepSuddenContr\\outputs\\fwdFacingStep_decomp\\dataPlusPhysics\\validators\\DP' + str(i) + '.vtp'])
    dP2vtp.PointArrayStatus = ['Re', 'Lo', 'Ho', 'true_p', 'true_u', 'true_v', 'pred_p', 'pred_u', 'pred_v']

    # Properties modified on dP2vtp
    dP2vtp.TimeArray = 'None'

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    dP2vtpDisplay = Show(dP2vtp, renderView1, 'GeometryRepresentation')

    # create a new 'P2_Delaunay'
    p2_Delaunay1 = P2_Delaunay(registrationName='P2_Delaunay1', Input=dP2vtp)

    # show data in view
    p2_Delaunay1Display = Show(p2_Delaunay1, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'diff_pressure'
    diff_pressureLUT = GetColorTransferFunction('diff_pressure')

    # trace defaults for the display properties.
    p2_Delaunay1Display.Representation = 'Surface'
    p2_Delaunay1Display.ColorArrayName = ['POINTS', 'diff_pressure']
    p2_Delaunay1Display.LookupTable = diff_pressureLUT
    p2_Delaunay1Display.SelectTCoordArray = 'None'
    p2_Delaunay1Display.SelectNormalArray = 'None'
    p2_Delaunay1Display.SelectTangentArray = 'None'
    p2_Delaunay1Display.OSPRayScaleArray = 'diff_pressure'
    p2_Delaunay1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    p2_Delaunay1Display.SelectOrientationVectors = 'diff_velocity'
    p2_Delaunay1Display.ScaleFactor = 1.8
    p2_Delaunay1Display.SelectScaleArray = 'diff_pressure'
    p2_Delaunay1Display.GlyphType = 'Arrow'
    p2_Delaunay1Display.GlyphTableIndexArray = 'diff_pressure'
    p2_Delaunay1Display.GaussianRadius = 0.09
    p2_Delaunay1Display.SetScaleArray = ['POINTS', 'diff_pressure']
    p2_Delaunay1Display.ScaleTransferFunction = 'PiecewiseFunction'
    p2_Delaunay1Display.OpacityArray = ['POINTS', 'diff_pressure']
    p2_Delaunay1Display.OpacityTransferFunction = 'PiecewiseFunction'
    p2_Delaunay1Display.DataAxesGrid = 'GridAxesRepresentation'
    p2_Delaunay1Display.PolarAxes = 'PolarAxesRepresentation'
    p2_Delaunay1Display.SelectInputVectors = ['POINTS', 'diff_velocity']
    p2_Delaunay1Display.WriteLog = ''

    # hide data in view
    Hide(dP2vtp, renderView1)

    # show color bar/color legend
    p2_Delaunay1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get opacity transfer function/opacity map for 'diff_pressure'
    diff_pressurePWF = GetOpacityTransferFunction('diff_pressure')

    # get 2D transfer function for 'diff_pressure'
    diff_pressureTF2D = GetTransferFunction2D('diff_pressure')
    
    
# create a new 'Plane'
plane2 = Plane(registrationName='Plane2')

# set active source
SetActiveSource(plane2)

# Properties modified on plane2
plane2.Origin = [0.0, 0.5, 0.0]
plane2.Point1 = [0.0, 0.16, 0.0]
plane2.Point2 = [12.0, 0.5, 0.0]

# show data in view
plane2Display = Show(plane2, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
plane2Display.Representation = 'Surface'

# change solid color
plane2Display.AmbientColor = [0.3215686274509804, 0.3411764705882353, 0.43137254901960786]
plane2Display.DiffuseColor = [0.3215686274509804, 0.3411764705882353, 0.43137254901960786]

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts
# layout/tab size in pixels
layout1.SetSize(2201, 1154)

#-----------------------------------
# saving camera placements for views
# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [3.0, 0.0, 34.826950949801585]
renderView1.CameraFocalPoint = [3.0, 0.0, 0.0]
renderView1.CameraParallelScale = 9.013878188659973

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
    
renderView1.Update()