from tkinter import *
import numpy
import math
import numpy as np
import datetime
# from noise import pnoise2
# from pynput.keyboard import Key, Listener


ticker = int(1)
# When ticker is at 261834, the normalisation breaks

def _from_rgb(rgb):
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'

def vec_divide(num, den):
	if den != 0:
		return vec3d(num.x/den, num.y/den, num.z/den)
	return num

def vec_multiply(num, k):
	return vec3d(num.x*k, num.y*k, num.z*k)

def vec_subtract(vec1, vec2):
	return vec3d(vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z)

def vec_add(vec1, vec2):
	return vec3d(vec1.x+vec2.x, vec1.y+vec2.y, vec1.z+vec2.z)

def mat_multiply(input1, input2):
	return np.matmul(input1,input2)

def mat_make_trans(x,y,z):
	sss = np.matrix([
		[1,0,0,0],
		[0,1,0,0],
		[0,0,1,0],
		[x,y,z,1],
		])
	return sss

def vec_dot_product(input1, input2):
	return input1.x * input2.x + input1.y * input2.y + input1.z * input2.z

def vec_length(input1):
	return sqrtf(vec_dot_product(v, v))

def vec_normalise(input1):
	l = vec_length(input1)
	return vec_divide(input1, l)

def vec_to_np(vec):
	return np.array([vec.x, vec.y, vec.z, ])

def np_to_vec(np):
	np=np.tolist()[0]
	return vec3d(np[0], np[1], np[2])

def vec_normal(vec):
		return math.sqrt(
			vec.x*vec.x+
			vec.y*vec.y+
			vec.z*vec.z)

def vec_cross_product(v1, v2):
		v = vec3d(v1.y * v2.z - v1.z * v2.y,
			v1.z * v2.x - v1.x * v2.z,
			v1.x * v2.y - v1.y * v2.x)
		return v


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
		global vCamera
		vCamera = vec3d(0,0,0)
		global fYaw
		fYaw = 0
		global fPitch
		fPitch = 0
		global fRoll
		fRoll = 0
		global vTarget
		vTarget = vec3d(0,0,1)


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
			self.canvas.create_polygon(coords, fill='blue', outline='black',tag=tag)
		self.canvas.pack()
	
	def refresh(self):
		#self.canvas.create_line(kwargs[0], kwargs[1], 300, 200, dash=(4, 2))
		#self.canvas.bind('<Motion>', self.mouse)
		self.mousex = self.winfo_pointerx()
		self.mousey = self.winfo_pointery()

		self.bind('<Key>', self.input)


		for i in objectsGlobal:
			triListTransformed = []
			for a in objectsGlobal[i].triList:
				triTransformed = []
				for corner in a.p:
					triTransformed.append(pipeline(corner))
				cp = cross_product(triTransformed)
				if cp[0]:
					triListTransformed.append([triTransformed, cp[1]])
			self.triSorted = self.depth_buffer_sort(triListTransformed)
			for a in self.triSorted:
					self.draw(a[0], i, a[1])

	def timed_refresh(self):
		# init_camera()
		global ticker
		for i in objectsGlobal:	
			self.canvas.delete(i)
		ticker += 1
		ticker_list.append(ticker)
		self.refresh()
		self.after(1, self.timed_refresh)

	def input(self, event):
		global vCamera
		global vLookDir
		global fYaw
		global fPitch
		global fRoll
		vForward = vec_multiply(vLookDir, 8)

		if event.char == 'w':
			vCamera.z += 0.1
		if event.char == 's':
			vCamera.z -= 0.1

		if event.char == 'a':
			vCamera.x += 0.1
		if event.char == 'd':
			vCamera.x -= 0.1

		# elif event.char == 'j':
		# 	vCamera.z += 0.1
		# elif event.char == 'k':
		# 	vCamera.z -= 0.1

		if event.char == 'i':
			vCamera = vec_add(vCamera, vForward)
		if event.char == 'k':
			vCamera = vec_subtract(vCamera, vForward)

		# if event.char == 'i':
		# 	fPitch += 0.1
		# if event.char == 'k':
		# 	fPitch -= 0.1

		if event.char == 'j':
			fYaw -= 0.1
		if event.char == 'l':
			fYaw += 0.1
		# print(f'{event.char}')


def light():
	light_direcion = vec3d(0,0,-1)
	## Normalise
	return vec_divide(light_direcion, vec_normal(light_direcion))

def pipeline(points):
	temp = vec_to_np(points)
	# temp = mat_multiply(temp, matRotX(fPitch))
	# temp = mat_multiply(temp, matRotY(fYaw))
	# temp = mat_multiply(temp, matRotZ(fRoll))

	# temp[2] = temp[2] + 30
	temp = np.append(np.array(temp), points.w)
	
	## World transformation
	temp = mat_multiply(temp, world_matrix())

	## View transformation
	temp = mat_multiply(temp, init_camera())

	## Projection transformation
	temp = mat_multiply(temp, projection_matrix())

	temp = temp.tolist()[0]
	projPoints = vec3d(temp[0], temp[1], temp[2])
	# projPoints = np_to_vec(temp)
	if temp[3] != 0:
		projPoints = vec_divide(projPoints, temp[3])
	return projPoints


def world_matrix():
	matTrans = mat_make_trans(0,0,15)
	matWorld = np.matrix([
		[1,0,0,0],
		[0,1,0,0],
		[0,0,1,0],
		[0,0,0,1],
		])
	# matWorld = mat_multiply(matRotX(ticker), matRotZ(ticker))
	matWorld = mat_multiply(matWorld, matTrans)
	return matWorld

def projection_matrix(
		screenwidth = 16,
		screenheight = 9,
		zfar = 1000.0,
		znear = 0.1,
		fov = 60.0,):
	znorm = zfar/(zfar-znear)
	aspectRatio = screenwidth/screenheight
	fovRad = 1/(math.tan((fov*0.5*math.pi)/(180.0)))
	angle = ticker/100

	matProj = np.matrix([
		[aspectRatio*fovRad,0,0,0],
		[0,fovRad,0,0],
		[0,0,znorm,1],
		[0,0,-(zfar-znear)/(zfar-znear),0]
		])
	return matProj

def matRotX(angle):
	matRotX = np.matrix([
		[1,0,0],
		[0,math.cos(angle),-(math.sin(angle))],
		[0,math.sin(angle),math.cos(angle)],
		])
	return matRotX

def matRotY(angle):
	matRotY = np.matrix([
		[math.cos(angle),0,math.sin(angle)],
		[0,1,0],
		[-(math.sin(angle)),0,math.cos(angle)],
		])
	return matRotY

def matRotZ(angle):
	matRotZ = np.matrix([
		[math.cos(angle),-(math.sin(angle)),0],
		[math.sin(angle),math.cos(angle),0],
		[0,0,1],
		])
	return matRotZ



def cross_product(tri):
	light_direcion = light()
	line1 = vec_subtract(tri[1], tri[0])
	line2 = vec_subtract(tri[2], tri[0])
	norm = vec_cross_product(line1, line2)
	# anormal = np.cross([line2.x, line1.y, line2.z],[[line2.x, line2.y, line2.z]]).tolist()
	# normal = vec3d(anormal[0], anormal[1], anormal[2])

	## Normalise
	normal = vec_divide(norm, vec_normal(norm))

	dp = vec_dot_product(normal, light_direcion)

	if (normal.x * (tri[0].x - vCamera.x) +
		normal.y * (tri[0].y - vCamera.y) +
		normal.z * (tri[0].z - vCamera.z) < 0.0):
		return [True, dp]
	else:
		return [False, 0]


def init_camera():
	global vCamera
	global vLookDir
	global vTarget
	vUp = vec3d(0,-1,0)
	# vLookDir = vec3d(0,0,1)
	matCameraRot = matRotY(fYaw)
	vLookDir = mat_multiply(matCameraRot, vec_to_np(vTarget))
	vLookDir = np_to_vec(vLookDir)
	vTarget = vec_add(vCamera, vLookDir)
	matCamera = mat_point_at(vCamera, vTarget, vUp)
	matView = np.linalg.inv(matCamera)
	return matView

def mat_point_at(pos, target, up):
	newForward = vec_subtract(target, up)
	## Normalise
	newForward = vec_divide(newForward, vec_normal(newForward))

	a = vec_multiply(newForward, vec_dot_product(up, newForward))
	newUp = vec_subtract(up, a)
	## Normalise
	newUp = vec_divide(newUp, vec_normal(newUp))

	newRight = vec_cross_product(newUp, newForward)
	Matrix = np.matrix([
				[newRight.x,newRight.y,newRight.z,0],
				[newUp.x,newUp.y,newUp.z,0],
				[newForward.x,newForward.y,newForward.z,0],
				[pos.x,pos.y,pos.z,1],
				])
	return Matrix

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
		# 'ship': object('D:\\Users\\Koope\\Desktop\\ship.obj', 'ship'),
		'axis': object('D:\\Users\\Koope\\Desktop\\axis.obj', 'axis'),
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
