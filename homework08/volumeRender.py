import vtk
import nibabel as nib
from vtkmodules.util import numpy_support

reduceSegment = True

### volume render
## 1: Source / Reader
## load files (e.g., .nii.gz) and convert them into vtk data structures
## file -> numpy array -> vtkImageData
file_path = "image_lr.nii.gz"
img = nib.load(file_path)
img_data = img.get_fdata()
dims = img.shape
# pixdim[1], pixdim[2], pixdim[3] represents the physical spacing in x, y, z
spacing = img.header["pixdim"][1: 4]

# create vtk image container
vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(dims[0], dims[1], dims[2])
vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])

# transfer (x, y, z) into (z, y, x) because the last index (x) changes the fastest
vtk_array = numpy_support.numpy_to_vtk(img_data.transpose(2, 1, 0).flatten(), deep=True)
# load data into spatial scalar field
vtk_image.GetPointData().SetScalars(vtk_array)

if reduceSegment:
    # use VTK native 3D Gaussian smoothing (High performance)
    filter = vtk.vtkImageGaussianSmooth()
    filter.SetInputData(vtk_image)
    # set standard deviation for smoothing (about 6\sigma \times 6\sigma pixels)
    filter.SetStandardDeviations(0.8, 0.8, 0.8)
    filter.Update()

## 2. construct transfer function 
volume_property = vtk.vtkVolumeProperty()
volume_property.ShadeOn() 
volume_property.SetInterpolationTypeToLinear() 
volume_property.SetDiffuse(0.8)
volume_property.SetSpecular(0.8)

opacity_func = vtk.vtkPiecewiseFunction()
opacity_func.AddPoint(20, 0.0)   
opacity_func.AddPoint(150, 0.2)  
opacity_func.AddPoint(500, 0.8)  
volume_property.SetScalarOpacity(opacity_func)

color_func = vtk.vtkColorTransferFunction()
color_func.AddRGBSegment(0, 0.0, 0.0, 0.0, 20, 0.2, 0.0, 0.0)    
color_func.AddRGBSegment(20, 0.1, 0.0, 0.0, 128, 1.0, 0.0, 0.0)    
volume_property.SetColor(color_func)

## 3. Mapper
## implement Ray Casting algorithm
## vtkImageData -> vtkVolumeRayCastMapper
mapper = vtk.vtkGPUVolumeRayCastMapper()
if reduceSegment:
    mapper.SetInputConnection(filter.GetOutputPort())
else:
    mapper.SetInputData(vtk_image)

## 4. Actor
## represent an object in the rendering scene with visual properties
## vtkVolumeRayCastMapper -> vtkVolume
actor = vtk.vtkVolume()
actor.SetMapper(mapper)
actor.SetProperty(volume_property)

## 5. Renderer
## manage the virtual scene including all actors
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
# white background
renderer.SetBackground(1, 1, 1)

## 6. Render Window
## provide a window on the operating system 
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 800)

## 7. Interactor
## enable user interaction
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

## final pipeline
# execute the pipeline to render the first frame
render_window.Render()

# save: window to image
# render: actor + renderer -> RenderWindow
# capture: RenderWindow -> vtkWindowToImageFilter
# output: vtkWindowToImageFilter -> vtkPNGWriter
w2i = vtk.vtkWindowToImageFilter()
w2i.SetInput(render_window) 
w2i.Update()

img_writer = vtk.vtkPNGWriter()
if reduceSegment:
    img_writer.SetFileName("reduceVolumeRender.png")
else:
    img_writer.SetFileName("volumeRender.png")
img_writer.SetInputConnection(w2i.GetOutputPort())
img_writer.Write()

# interacte
interactor.Initialize()
interactor.Start()