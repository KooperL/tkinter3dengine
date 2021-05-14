from tkinter import *
import numpy
import math
import numpy as np
import datetime
# from noise import pnoise2

ticker = int(1)
# When ticker is at 261834, the normalisation breaks

def _from_rgb(rgb):
    #translates an rgb tuple of int to a tkinter friendly colour code
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'

def vec_divide(num, den):
	return num.x/den, num.y/den, num.z/den


class vec3d:
	def __init__(self, x=0, y=0, z=0):
		self.x, self.y, self.z = x, y, z

class triangle:
	def __init__(self, v1, v2, v3):
		self.p=[v1, v2, v3]

class object:
	def __init__(self, objdir, name, file=1):
		self.name = name
		if file == 1:
			vlist = open(objdir, 'r').readlines()
			self.meshObj(vlist)
		else:
			self.mesh(self.vlist)

	def mesh(self, rawPoints):
		self.triList = []
		for point in rawPoints:
			a = triangle(
				vec3d(point[0][0], point[0][1], point[0][2]),
				vec3d(point[1][0], point[1][1], point[1][2]),
				vec3d(point[2][0], point[2][1], point[2][2]))
			self.triList.append(a)
		return

	def meshObj(self, rawPoints):
		self.triList = []
		self.verts = []
		for line in rawPoints:
			line = line.split()
			if not line:
				next
			elif line[0] == 'v':
				self.verts.append(vec3d(float(line[1]), float(line[2]), float(line[3])))
			elif line[0] == 'f':				
					self.triList.append(triangle(
						vec3d(self.verts[int(line[1])-1].x, self.verts[int(line[1])-1].y, self.verts[int(line[1])-1].z),
						vec3d(self.verts[int(line[2])-1].x, self.verts[int(line[2])-1].y, self.verts[int(line[2])-1].z),
						vec3d(self.verts[int(line[3])-1].x, self.verts[int(line[3])-1].y, self.verts[int(line[3])-1].z)))



class Window(Tk):
	def __init__(self):
		Tk.__init__(self)

		# self.meshCubeObj = object(meshCube, 'test', 0)
		self.ship = object('D:\\Users\\Koope\\Desktop\\ship.obj', 'ship')

		self.screen_width = self.winfo_screenwidth()
		self.screen_height = self.winfo_screenheight()
		self.title("Tkinter window")
		self.geometry("%dx%d" % (self.screen_width, self.screen_height))
		self.canvas = Canvas(self, width=self.screen_width, height=self.screen_height)
		self.canvas.create_rectangle(0, 0, self.screen_width, self.screen_height, fill='gray', outline='gray')
		self.wait_visibility()
		self.timed_refresh()

	def depth_buffer_sort(self, points):
		self.sortedPoints = []
		for i in range(0,len(points),2):
			z1 = points[i-1][0][0].z+points[i-1][0][1].z+points[i-1][0][2].z/3
			z2 = points[i][0][0].z+points[i][0][1].z+points[i][0][2].z/3
			if z1>z2:
				self.sortedPoints.append([points[i-1][0], points[i-1][1]])
				self.sortedPoints.append([points[i][0], points[i][1]])
			else:
				self.sortedPoints.append([points[i][0], points[i][1]])
				self.sortedPoints.append([points[i-1][0], points[i-1][1]])
		return self.sortedPoints

	def draw(self, points, tag, shade):
		# print(f'Passed args: {kwargs}')
		col = _from_rgb((
			int(100**(shade)),
			int(100**(shade)),
			int(100**(shade))))

		for i in [points]:
			xScale, yScale = 0.25, 0.5
			xTransform, yTransform = self.screen_width/3, self.screen_height/3
			coords = []
			for e in i:
				coords.append(e.x*xScale*self.screen_width+xTransform)
				coords.append(e.y*yScale*self.screen_height+yTransform)
			self.canvas.create_polygon(coords, fill=col, outline='',tag=tag)
		self.canvas.pack()
	
	def refresh(self):
		#self.canvas.create_line(kwargs[0], kwargs[1], 300, 200, dash=(4, 2))
		#self.canvas.bind('<Motion>', self.mouse)
		self.mousex = self.winfo_pointerx()
		self.mousey = self.winfo_pointery()

		###
		###   Hard coded objects
		###

		# for tri in self.meshCubeObj.triList:
		self.triListTransformed = []
		for tri in self.ship.triList:
			self.triTransformed = []
			for corner in tri.p:
				self.triTransformed.append(projection_matrix(corner))
			cp = cross_product(self.triTransformed)
			if cp[0]:
				self.triListTransformed.append([self.triTransformed, cp[1]])
		# print(self.triListTransformed)
		self.triSorted = self.depth_buffer_sort(self.triListTransformed)
		for i in self.triSorted:
				self.draw(i[0], 'meshCube', i[1])

	def timed_refresh(self):
		init_camera()
		global ticker
		self.canvas.delete('meshCube')
		ticker += 1
		ticker_list.append(ticker)
		self.refresh()
		self.after(1, self.timed_refresh)



def light():
	light_direcion = vec3d(0,0,-1)
	l = math.sqrt(light_direcion.x*light_direcion.x+light_direcion.y*light_direcion.y+light_direcion.z*light_direcion.z)
	light_direcion.x /= l
	light_direcion.y /= l
	light_direcion.z /= l
	return light_direcion

def projection_matrix(points):
	screenwidth = 16
	screenheight = 9
	zfar = 1000.0
	znear = 0.1
	fov = 90.0
	znorm = zfar/(zfar-znear)
	aspectRatio = screenwidth/screenheight
	fovRad = 1/(math.tan((fov*0.5*math.pi)/(180.0)))

		## Transformations here
	points = rotateX_matrix(points)
	# points = rotateY_matrix(points)
	points = rotateZ_matrix(points)

	trianglePoints = np.matrix([points.x,points.y,points.z+15,1])
	matProj = np.matrix([
				[aspectRatio*fovRad,0,0,0],
				[0,fovRad,0,0],
				[0,0,znorm,1],
				[0,0,-(zfar-znear)/(zfar-znear),0]
			])
	projPointsRaw = (numpy.matmul(trianglePoints,matProj)).tolist()[0]
	if projPointsRaw[3] != 0:
		projPoints = vec3d(projPointsRaw[0]/projPointsRaw[3], projPointsRaw[1]/projPointsRaw[3], projPointsRaw[2]/projPointsRaw[3])
	else:
		projPoints = vec3d(projPointsRaw[0], projPointsRaw[1], projPointsRaw[2])
	return projPoints


## subtracy
def cross_product(tri):
	light_direcion = light()
	line1 = vec3d(
		tri[1].x-tri[0].x, 
		tri[1].y-tri[0].y, 
		tri[1].z-tri[0].z)
	line2 = vec3d(
		tri[2].x-tri[0].x, 
		tri[2].y-tri[0].y, 
		tri[2].z-tri[0].z)
	normal = vec3d(
		line1.y * line2.z - line1.z * line2.y,
		line1.z * line2.x - line1.x * line2.z,
		line1.x * line2.y - line1.y * line2.x)
	# anormal = np.cross([line2.x, line1.y, line2.z],[[line2.x, line2.y, line2.z]]).tolist()
	# normal = vec3d(anormal[0], anormal[1], anormal[2])

## Normalise function?
	l = math.sqrt(normal.x*normal.x+normal.y*normal.y+normal.z*normal.z)
	normal.x /= l
	normal.y /= l
	normal.z /= l

	#inefficient
	dp = normal.x * light_direcion.x + normal.y * light_direcion.y + normal.z * light_direcion.z

	if (normal.x * (tri[0].x - vcamera.x) +
		normal.y * (tri[0].y - vcamera.y) +
		normal.z * (tri[0].z - vcamera.z) < 0.0):
		return [True, dp]
	else:
		return [False, 0]

# def dotPproduct(normals, translateds, camera):


def init_camera():
	global vcamera
	vcamera = vec3d(0,0,0)
	# vLookDir = [0,0,1]
	# vUp = [0,1,0]
	# vTarget = np.add(vTarget, vUp)

def camera_matrix(points):
	pPoints = np.matrix([point[0],point[1],point[2],1])
	pointAtMatrix = np.matrix([
				[Ax,Ay,Az,0],
				[Bx,By,Bz,0],
				[Cx,Cy,Cz,0],
				[Tx,Ty,Tz,1],
				])
	lookAtMatrix = np.linalg.inv(pointAtMatrix)
	return (numpy.matmul(np.matrix(points),matRotY)).tolist()[0]


def rotateX_matrix(points):
	angle = ticker/100
	matRotX = np.matrix([
		[1,0,0],
		[0,math.cos(angle),-(math.sin(angle))],
		[0,math.sin(angle),math.cos(angle)],
		])
	lst = np.matrix([points.x, points.y, points.z])
	x = (numpy.matmul(lst,matRotX)).tolist()[0]
	return vec3d(x[0], x[1], x[2])

def rotateY_matrix(points):
	angle = ticker/100
	matRotY = np.matrix([
		[math.cos(angle),0,math.sin(angle)],
		[0,1,0],
		[-(math.sin(angle)),0,math.cos(angle)],
		])
	lst = np.matrix([points.x, points.y, points.z])
	y = (numpy.matmul(lst,matRotY)).tolist()[0]
	return vec3d(y[0], y[1], y[2])

def rotateZ_matrix(points):
	angle = ticker/100
	matRotZ = np.matrix([
		[math.cos(angle),-(math.sin(angle)),0],
		[math.sin(angle),math.cos(angle),0],
		[0,0,1],
		])
	lst = np.matrix([points.x, points.y, points.z])
	z = (numpy.matmul(lst,matRotZ)).tolist()[0]
	return vec3d(z[0], z[1], z[2])

# def perlin_array(shape = (80, 80),
# 			scale=100, octaves = 12,
# 			persistence = 0.025, 
# 			lacunarity = 2.0, 
# 			seed = None):
#     global ready
#     if not seed:
#         seed = np.random.randint(0, 100)
#         print("seed was {}".format(seed))
#     arr = np.zeros(shape)
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             arr[i][j] = pnoise2(i / scale,
#                                         j / scale,
#                                         octaves=octaves,
#                                         persistence=persistence,
#                                         lacunarity=lacunarity,
#                                         repeatx=1024,
#                                         repeaty=1024,
#                                         base=seed)
#     return arr

def main():
	app = Window()
	app.mainloop()

tripoints = [[50, 50, -500, 0],[100, 50, 0, 0],[100, 100, 0, 0]]#,[450, 500, 0]

meshCube = [
	[[ 0.0, 0.0, 0.0,], [0.0, 1.0, 0.0,], [1.0, 1.0, 0.0 ]],
	[[ 0.0, 0.0, 0.0,], [1.0, 1.0, 0.0,], [1.0, 0.0, 0.0 ]],

	[[ 1.0, 0.0, 0.0,], [1.0, 1.0, 0.0,], [1.0, 1.0, 1.0 ]],
	[[ 1.0, 0.0, 0.0,], [1.0, 1.0, 1.0,], [1.0, 0.0, 1.0 ]],

	[[ 1.0, 0.0, 1.0,], [1.0, 1.0, 1.0,], [0.0, 1.0, 1.0 ]],
	[[ 1.0, 0.0, 1.0,], [0.0, 1.0, 1.0,], [0.0, 0.0, 1.0 ]],

	[[ 0.0, 0.0, 1.0,], [0.0, 1.0, 1.0,], [0.0, 1.0, 0.0 ]],
	[[ 0.0, 0.0, 1.0,], [0.0, 1.0, 0.0,], [0.0, 0.0, 0.0 ]],

	[[ 0.0, 1.0, 0.0,], [0.0, 1.0, 1.0,], [1.0, 1.0, 1.0 ]],
	[[ 0.0, 1.0, 0.0,], [1.0, 1.0, 1.0,], [1.0, 1.0, 0.0 ]],

	[[ 1.0, 0.0, 1.0,], [0.0, 0.0, 1.0,], [0.0, 0.0, 0.0 ]],
	[[ 1.0, 0.0, 1.0,], [0.0, 0.0, 0.0,], [1.0, 0.0, 0.0 ]],
]

if __name__ == '__main__':
	ticker_list = []
	start = datetime.datetime.now()
	main()
	end = datetime.datetime.now()
	secselapsed = end-start
	fps = len(ticker_list)/secselapsed.total_seconds()
	print(fps)

	## funct to load objects

# from pynput.keyboard import Key, Listener

# def on_press1(key):  
#     print('{0} pressed'.format(key))

# def on_release1(key):
#     print('{0} release'.format(key))
#     if key == Key.esc:
#         # Stop listener
#         return False

# # Collect events until released
# with Listener(
#         on_press=on_press1,
#         on_release=on_release1) as listener:
#     listener.join()
