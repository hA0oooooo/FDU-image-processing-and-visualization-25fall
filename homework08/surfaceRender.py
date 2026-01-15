import vtk
import nibabel as nib
from vtkmodules.util import numpy_support
import numpy as np

reduceSegment = True

### surface render
## 1: Source / Reader
## load files (e.g., .nii.gz) and convert them into vtk data structures
## file -> numpy array -> vtkImageData
file_path = "image_lr.nii.gz"
img = nib.load(file_path)
img_data = img.get_fdata()
dims = img.shape
# pixdim[1], pixdim[2], pixdim[3] represents the physical spacing in x, y, z
spacing = img.header["pixdim"][1: 4]

# manual realization for mean filter    
# def d3meanFilter(data):
#     dim = data.shape
#     padded = np.pad(array=data, pad_width=1, mode="constant", constant_values=0)
#     res = np.zeros_like(data)
#     for dx in range(3):
#         for dy in range(3):
#             for dz in range(3):
#                 res += padded[dx: dx+dim[0], dy: dy+dim[1], dz: dz+dim[2]]
#     return res / 27.0
# img_data = d3meanFilter(img_data)

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

## 2. Filter: turn volume data into surface
## apply geometric algorithms to extract or modify data
## vtkImageData -> vtkPolyData
if reduceSegment:
    extractor = vtk.vtkMarchingCubes()
    extractor.SetInputConnection(filter.GetOutputPort())
else:
    extractor = vtk.vtkMarchingCubes()
    extractor.SetInputData(vtk_image)
extractor.SetValue(0, 150)

# create triangle strips
stripper = vtk.vtkStripper()
stripper.SetInputConnection(extractor.GetOutputPort())

## 3. Mapper
## map geometry into graphics that the rendering engine can understand
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(stripper.GetOutputPort())
mapper.ScalarVisibilityOff()

## 4. Actor
## represent an object in the rendering scene with visual properties
actor = vtk.vtkActor()
actor.SetMapper(mapper)
# yellow foreground
actor.GetProperty().SetColor(1, 1, 0)
actor.GetProperty().SetOpacity(0.95)
# mirror reflection intensity
actor.GetProperty().SetSpecular(1.0)

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
    img_writer.SetFileName("reducedSurfaceRender.png")
else:
    img_writer.SetFileName("surfaceRender.png")
img_writer.SetInputConnection(w2i.GetOutputPort())
img_writer.Write()

# interacte
interactor.Initialize()
interactor.Start()