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

def multiply_vecmat(vec, mat):
	# print(mat)
	if mat.shape[0] == 1:
		mat=mat.tolist()[0]
	else:
		mat=mat.tolist()
	# print(mat)
	return vec3d(
		vec.x*mat[0][0]+vec.y*mat[1][0]+vec.z*mat[2][0]+vec.w*mat[3][0], 
		vec.x*mat[0][1]+vec.y*mat[1][1]+vec.z*mat[2][1]+vec.w*mat[3][1], 
		vec.x*mat[0][2]+vec.y*mat[1][2]+vec.z*mat[2][2]+vec.w*mat[3][2],
		vec.x*mat[0][3]+vec.y*mat[1][3]+vec.z*mat[2][3]+vec.w*mat[3][3])

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

def vec_to_np(vec, w=1):
	# if w == 1:
	# 	return np.array([vec.x, vec.y, vec.z, vec.w])
	return np.array([vec.x, vec.y, vec.z, vec.w])

def np_to_vec(np):
	# print(np.shape)
	np=np.tolist()
	# print(np)
	if np[3]:
		return vec3d(np[0], np[1], np[2], np[3])
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
		global light_direcion
		light_direcion = vec3d(0,0,-1)
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
		xScale, yScale = 0.25, 0.5
		xTransform, yTransform = self.screen_width/3, self.screen_height/3
		self.canvas.create_polygon(
			points[0].x*xScale*self.screen_width+xTransform,
			points[0].y*yScale*self.screen_height+yTransform,
			points[1].x*xScale*self.screen_width+xTransform,		
			points[1].y*yScale*self.screen_height+yTransform,
			points[2].x*xScale*self.screen_width+xTransform,
			points[2].y*yScale*self.screen_height+yTransform,
			fill=col, outline='black',tag=tag)
		self.canvas.pack()
	
	def timed_refresh(self):
		global ticker
		for i in objectsGlobal:	
			self.canvas.delete(i)
		ticker += 1
		self.pipeline()
		# print(f'fYaw: {fYaw}, fPitch: {fPitch}')
		# print(f'x: {vCamera.x}, y {vCamera.y}, z: {vCamera.z}')
		self.after(1, self.timed_refresh)

	def press(self, event):
		global vCamera
		global vLookDir
		global fYaw
		global fPitch
		global fRoll
		vForward = vec_multiply(vLookDir, 0.0001)

		if event.char == 'w':
			vCamera.z += 1
		if event.char == 's':
			vCamera.z -= 1

		if event.char == 'a':
			vCamera.x += 1
		if event.char == 'd':
			vCamera.x -= 1

		elif event.char == 'q':
			vCamera.y += 0.1
		elif event.char == 'e':
			vCamera.y -= 0.1

		# if event.char == 'w':
		# 	vCamera = vec_add(vCamera, vForward)
		# if event.char == 's':
		# 	vCamera = vec_subtract(vCamera, vForward)

		if event.char == 'i':
			fPitch += 0.01
		if event.char == 'k':
			fPitch -= 0.01

		if event.char == 'j':
			fYaw += 0.01
		if event.char == 'l':
			fYaw -= 0.01
		# print(f'{event.char}')

	def release(self, event):
		global vCamera
		global vLookDir
		global fYaw
		global fPitch
		global fRoll
		vForward = vec_multiply(vLookDir, 0.0001)

		if event.char == 'w':
			vCamera.z -= 1
		if event.char == 's':
			vCamera.z += 1

		if event.char == 'a':
			vCamera.x -= 1
		if event.char == 'd':
			vCamera.x += 1

		elif event.char == 'q':
			vCamera.y -= 0.1
		elif event.char == 'e':
			vCamera.y += 0.1

		# if event.char == 'w':
		# 	vCamera = vec_add(vCamera, vForward)
		# if event.char == 's':
		# 	vCamera = vec_subtract(vCamera, vForward)

		if event.char == 'i':
			fPitch -= 0.01
		if event.char == 'k':
			fPitch += 0.01

		if event.char == 'j':
			fYaw -= 0.01
		if event.char == 'l':
			fYaw += 0.01
		# print(f'{event.char}')

	def pipeline(self):
		#self.canvas.create_line(kwargs[0], kwargs[1], 300, 200, dash=(4, 2))
		#self.canvas.bind('<Motion>', self.mouse)
		self.mousex = self.winfo_pointerx()
		self.mousey = self.winfo_pointery()
		self.bind('<KeyPress>', self.press)
		# self.bind('<KeyRelease>', self.release)
		for obj in objectsGlobal:
			triListTransformed = []
			for corner in objectsGlobal[obj].triList:

				## World transformation
				triTransformed = triangle(
					multiply_vecmat(corner.p[0], world_matrix()),
					multiply_vecmat(corner.p[1], world_matrix()),
					multiply_vecmat(corner.p[2], world_matrix()))

				line1 = vec_subtract(triTransformed.p[1], triTransformed.p[0])
				line2 = vec_subtract(triTransformed.p[2], triTransformed.p[0])
				norm = vec_cross_product(line1, line2)
				# anormal = np.cross([line2.x, line1.y, line2.z],[[line2.x, line2.y, line2.z]]).tolist()
				# normal = vec3d(anormal[0], anormal[1], anormal[2])

				## Normalise
				l = vec_normal(norm)
				normal = vec_divide(norm, l)

				if (normal.x * (triTransformed.p[0].x - vCamera.x) +
					normal.y * (triTransformed.p[0].y - vCamera.y) +
					normal.z * (triTransformed.p[0].z - vCamera.z) < 0.0):

					l = vec_normal(light_direcion)
					l = vec_divide(light_direcion, l)
					dp = vec_dot_product(normal, l)
				else:
					continue
				

				## View transformation
				triViewed = triangle(
					multiply_vecmat(triTransformed.p[0], init_camera()),
					multiply_vecmat(triTransformed.p[1], init_camera()),
					multiply_vecmat(triTransformed.p[2], init_camera()))


				## Clipping occurs here

				## Projection transformation
				triProjected = triangle(
					multiply_vecmat(triViewed.p[0], projection_matrix()),
					multiply_vecmat(triViewed.p[1], projection_matrix()),
					multiply_vecmat(triViewed.p[2], projection_matrix()))

				triProjected.p[0] = vec_divide(triProjected.p[0], triProjected.p[0].w)
				triProjected.p[1] = vec_divide(triProjected.p[1], triProjected.p[1].w)
				triProjected.p[2] = vec_divide(triProjected.p[2], triProjected.p[2].w)
				# self.triSorted = self.depth_buffer_sort(triListTransformed)
				self.draw(triProjected.p, obj, dp)


def world_matrix():
	matTrans = mat_make_trans(0,0,5)
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
	# angle = ticker/100

	matProj = np.matrix([
		[aspectRatio*fovRad,0,0,0],
		[0,fovRad,0,0],
		[0,0,znorm,1],
		[0,0,-(zfar-znear)/(zfar-znear),0]
		])
	return matProj

def matRotX(angle):
	matRotX = np.matrix([
		[1,0,0,0],
		[0,math.cos(angle),-(math.sin(angle)),0],
		[0,math.sin(angle),math.cos(angle),0],
		[0,0,0,1]
		])
	return matRotX

def matRotY(angle):
	matRotY = np.matrix([
		[math.cos(angle),0,math.sin(angle),0],
		[0,1,0,0],
		[-(math.sin(angle)),0,math.cos(angle),0],
		[0,0,0,1]
		])
	return matRotY

def matRotZ(angle):
	matRotZ = np.matrix([
		[math.cos(angle),-(math.sin(angle)),0],
		[math.sin(angle),math.cos(angle),0],
		[0,0,1],
		])
	return matRotZ


def init_camera():
	global vCamera
	global vLookDir
	global vTarget
	global fYaw
	global fPitch
	print(f'vTarget: {vTarget.x},{vTarget.y},{vTarget.z}')
	vUp = vec3d(0,-1,0)
	vLookDir = multiply_vecmat(vTarget, matRotY(fYaw))
	# vLookDir = multiply_vecmat(vLookDir, matRotX(fPitch))
	fYaw = 0
	fPitch = 0
	vTarget = vec_add(vCamera, vLookDir)
	matCamera = mat_point_at(vCamera, vTarget, vUp)
	matView = np.linalg.inv(matCamera) ## Crash here if something.z == 0
	# print(f'vLookDir: {vLookDir.x},{vLookDir.y},{vLookDir.z}')
	# print(f'vUp: {vUp.x},{vUp.y},{vUp.z}, vTarget: {vTarget.x},{vTarget.y},{vTarget.z}')
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


def main():
	global objectsGlobal
	objectsGlobal = {
		# 'ship': object('D:\\Users\\Koope\\Desktop\\ship.obj', 'ship'),
		'axis': object('D:\\Users\\Koope\\Desktop\\axis.obj', 'axis'),
		}

	App = Window()
	App.mainloop()


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
	start = datetime.datetime.now()
	main()
	end = datetime.datetime.now()
	secselapsed = end-start
	fps = ticker/secselapsed.total_seconds()
	print(f'fps: {int(fps)}')
