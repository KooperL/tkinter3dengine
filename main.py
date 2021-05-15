from tkinter import *
import numpy
import math
import numpy as np
import datetime
# from noise import pnoise2

ticker = int(1)
# When ticker is at 261834, the normalisation breaks

def _from_rgb(rgb):
    #translates an rgb tuple of int to a tkinter friendly color code
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'

def vec_divide(num, den):
	if den != 0:
		return vec3d(num.x/den, num.y/den, num.z/den)
	return num

def vec_subtract(vec1, vec2):
	return vec3d(vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z)

def mat_multiply(input1, input2):
	return (numpy.matmul(input1,input2))

def vec_normalise(vec):
		return math.sqrt(
			vec.x*vec.x+
			vec.y*vec.y+
			vec.z*vec.z)

class vec3d:
	def __init__(self, x=0, y=0, z=0, w=1):
		self.x, self.y, self.z, self.w = x, y, z, w

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
			z1 = (points[i-1][0][0].z+points[i-1][0][1].z+points[i-1][0][2].z)/3
			z2 = (points[i][0][0].z+points[i][0][1].z+points[i][0][2].z)/3
			if z1>z2:
				self.sortedPoints.append([points[i-1][0], points[i-1][1]])
				self.sortedPoints.append([points[i][0], points[i][1]])
				next
			self.sortedPoints.append([points[i][0], points[i][1]])
			self.sortedPoints.append([points[i-1][0], points[i-1][1]])
		return self.sortedPoints

	def draw(self, points, tag, shade):
		# print(f'Passed args: {points}, {tag}, {shade})')
		col = _from_rgb((
			int(100**shade),
			int(100**shade),
			int(100**shade)))

		for point in [points]:
			xScale, yScale = 0.25, 0.5
			xTransform, yTransform = self.screen_width/3, self.screen_height/3
			coords = []
			for xyz in point:
				coords.append(xyz.x*xScale*self.screen_width+xTransform)
				coords.append(xyz.y*yScale*self.screen_height+yTransform)
			self.canvas.create_polygon(coords, fill=col, outline='black',tag=tag)
		self.canvas.pack()
	
	def refresh(self):
		#self.canvas.create_line(kwargs[0], kwargs[1], 300, 200, dash=(4, 2))
		#self.canvas.bind('<Motion>', self.mouse)
		self.mousex = self.winfo_pointerx()
		self.mousey = self.winfo_pointery()

		for i in objectsGlobal:
			triListTransformed = []
			for a in objectsGlobal[i].triList:
				triTransformed = []
				for corner in a.p:
					triTransformed.append(projection_matrix(corner))
				cp = cross_product(triTransformed)
				if cp[0]:
					triListTransformed.append([triTransformed, cp[1]])
			self.triSorted = self.depth_buffer_sort(triListTransformed)
			for a in self.triSorted:
					self.draw(a[0], i, a[1])

	def timed_refresh(self):
		init_camera()
		global ticker
		for i in objectsGlobal:	
			self.canvas.delete(i)
		ticker += 1
		ticker_list.append(ticker)
		self.refresh()
		self.after(1, self.timed_refresh)


def light():
	light_direcion = vec3d(0,0,-1)
	return vec_divide(light_direcion, vec_normalise(light_direcion))

#Object update
def projection_matrix(points,
		screenwidth = 16,
		screenheight = 9,
		zfar = 1000.0,
		znear = 0.1,
		fov = 90.0,):
	znorm = zfar/(zfar-znear)
	aspectRatio = screenwidth/screenheight
	fovRad = 1/(math.tan((fov*0.5*math.pi)/(180.0)))

	## Transformations here
	angle = ticker/100
	temp = np.matrix([points.x, points.y, points.z])

	matRotX = np.matrix([
		[1,0,0],
		[0,math.cos(angle),-(math.sin(angle))],
		[0,math.sin(angle),math.cos(angle)],
		])

	matRotY = np.matrix([
		[math.cos(angle),0,math.sin(angle)],
		[0,1,0],
		[-(math.sin(angle)),0,math.cos(angle)],
		])

	matRotZ = np.matrix([
		[math.cos(angle),-(math.sin(angle)),0],
		[math.sin(angle),math.cos(angle),0],
		[0,0,1],
		])

	matProj = np.matrix([
		[aspectRatio*fovRad,0,0,0],
		[0,fovRad,0,0],
		[0,0,znorm,1],
		[0,0,-(zfar-znear)/(zfar-znear),0]
		])

	temp = mat_multiply(temp, matRotX)
	temp = mat_multiply(temp, matRotY)
	temp = mat_multiply(temp, matRotZ)
	
	temp = np.append(np.array(temp), points.w)
	temp[2] = temp[2] + 15
	temp = mat_multiply(temp, matProj).tolist()[0]
	projPoints = vec3d(temp[0], temp[1], temp[2])

	if temp[3] != 0:
		projPoints = vec_divide(projPoints, temp[3])
	return projPoints


def cross_product(tri):
	light_direcion = light()
	line1 = vec_subtract(tri[1], tri[0])
	line2 = vec_subtract(tri[2], tri[0])
	norm = vec3d(
		line1.y * line2.z - line1.z * line2.y,
		line1.z * line2.x - line1.x * line2.z,
		line1.x * line2.y - line1.y * line2.x)
	# anormal = np.cross([line2.x, line1.y, line2.z],[[line2.x, line2.y, line2.z]]).tolist()
	# normal = vec3d(anormal[0], anormal[1], anormal[2])

	normal = vec_divide(norm, vec_normalise(norm))

	#inefficient multi
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
	global objectsGlobal
	objectsGlobal = {
		'ship': object('D:\\Users\\Koope\\Desktop\\ship.obj', 'ship'),
		}

	app = Window()
	app.mainloop()


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
	print(f'fps: {int(fps)}')

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
