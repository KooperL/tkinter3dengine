from tkinter import *
import numpy
import math
import numpy as np
import datetime
# from noise import pnoise2
# from pynput.keyboard import Key, Listener


ticker = int(1)
# When ticker is at 261834, the normalisation breaks

class vec3d:
	def __init__(self, 
		x=0, 
		y=0, 
		z=0, 
		w=1):
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)
		self.w = float(w)

class triangle:
	def __init__(self,
		v1=vec3d(),
		v2=vec3d(),
		v3=vec3d(),
		dp=None):
		self.p=[v1, v2, v3]
		self.dp=dp
		self.col=None

class object:
	def __init__(self, objdir, name, vCoords=(0,0,0) , file=1):
		self.name = name
		self.vCoords = vCoords
		if file == 1:
			vlist = open(objdir, 'r').readlines()
			self.type1(vlist)
		else:
			self.type2(vlist)

	def type1(self, rawPoints):
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

	def type2(self, rawPoints):
		pass



def vector_intersect_plane(plane_p: vec3d, plane_n: vec3d, lineStart: vec3d, lineEnd: vec3d) -> vec3d:
		plane_n = vec_normalise(plane_n)
		plane_d = -vec_dot_product(plane_n, plane_p)
		ad = vec_dot_product(lineStart, plane_n)
		bd = vec_dot_product(lineEnd, plane_n)
		t = (-plane_d - ad) / (bd - ad)
		lineStartToEnd = vec_subtract(lineEnd, lineStart)
		lineToIntersect = vec_multiply(lineStartToEnd, t)
		return vec_add(lineStart, lineToIntersect)

def Triangle_ClipAgainstPlane(plane_p: vec3d, plane_n: vec3d, in_tri: triangle, out_tri1: triangle, out_tri2: triangle) -> int:
		plane_n = vec_normalise(plane_n)

		def dist(p: vec3d):
			n = vec_normalise(p)
			return (plane_n.x * p.x + plane_n.y * p.y + plane_n.z * p.z - vec_dot_product(plane_n, plane_p))

		inside_points = [0,0,0]
		nInsidePointCount = int(0)
		outside_points = [0,0,0]
		nOutsidePointCount = int(0)

		d0 = float(dist(in_tri.p[0]))
		d1 = float(dist(in_tri.p[1]))
		d2 = float(dist(in_tri.p[2]))

		if (d0 >= 0):
			inside_points[nInsidePointCount] = in_tri.p[0]
			nInsidePointCount+=1
		else:
			outside_points[nOutsidePointCount] = in_tri.p[0]
			nOutsidePointCount+=1
		if (d1 >= 0):
			inside_points[nInsidePointCount] = in_tri.p[1]
			nInsidePointCount+=1
		else:
			outside_points[nOutsidePointCount] = in_tri.p[1]
			nOutsidePointCount+=1
		if (d2 >= 0):
			inside_points[nInsidePointCount] = in_tri.p[2]
			nInsidePointCount+=1
		else:
			outside_points[nOutsidePointCount] = in_tri.p[2]
			nOutsidePointCount+=1

		if (nInsidePointCount == 0):
			return [0]

		if (nInsidePointCount == 3):
			out_tri1 = in_tri
			return 1, out_tri1

		if (nInsidePointCount == 1 and nOutsidePointCount == 2):
			out_tri1.col =  in_tri.col

			# The inside point is valid, so keep that...
			out_tri1.p[0] = inside_points[0]
			out_tri1.p[1] = vector_intersect_plane(plane_p, plane_n, inside_points[0], outside_points[0])
			out_tri1.p[2] = vector_intersect_plane(plane_p, plane_n, inside_points[0], outside_points[1])
			return 1, out_tri1 # Return the newly formed single triangle

		if (nInsidePointCount == 2 and nOutsidePointCount == 1):

			out_tri1.col =  in_tri.col
			out_tri2.col =  in_tri.col

			out_tri1.p[0] = inside_points[0]
			out_tri1.p[1] = inside_points[1]
			out_tri1.p[2] = vector_intersect_plane(plane_p, plane_n, inside_points[0], outside_points[0])

			out_tri2.p[0] = inside_points[1]
			out_tri2.p[1] = out_tri1.p[2]
			out_tri2.p[2] = vector_intersect_plane(plane_p, plane_n, inside_points[1], outside_points[0])
			return 2, out_tri1, out_tri2


def getZs(tri):
	#lambda tri: (tri.p[0].z+tri.p[1].z+tri.p[2].z)/3
	#np.mean(tri.p[0].z+tri.p[1].z+tri.p[2].z, dtype=float64)
	return (tri.p[0].z+tri.p[1].z+tri.p[2].z)/3

def _from_rgb(rgb):
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'

def vec_divide(num: vec3d, den: int) -> vec3d:
	if den != 0:
		return vec3d(num.x/den, num.y/den, num.z/den)
	return num

def vec_multiply(num: vec3d, k: int) -> vec3d:
	return vec3d(num.x*k, num.y*k, num.z*k)

def multiply_vecmat(vec: vec3d, mat: np.matrix) -> vec3d:
	if mat.shape[0] == 1:
		mat=mat.tolist()[0]
	else:
		mat=mat.tolist()
	return vec3d(
		vec.x*mat[0][0]+vec.y*mat[1][0]+vec.z*mat[2][0]+vec.w*mat[3][0], 
		vec.x*mat[0][1]+vec.y*mat[1][1]+vec.z*mat[2][1]+vec.w*mat[3][1], 
		vec.x*mat[0][2]+vec.y*mat[1][2]+vec.z*mat[2][2]+vec.w*mat[3][2],
		vec.x*mat[0][3]+vec.y*mat[1][3]+vec.z*mat[2][3]+vec.w*mat[3][3])

def vec_subtract(vec1: vec3d, vec2: vec3d) -> vec3d:
	return vec3d(vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z)

def vec_add(vec1: vec3d, vec2: vec3d) -> vec3d:
	return vec3d(vec1.x+vec2.x, vec1.y+vec2.y, vec1.z+vec2.z)

def mat_multiply(input1, input2):
	return np.matmul(input1,input2)

def mat_make_trans(x,y,z) -> np.matrix:
	return np.matrix([
		[1,0,0,0],
		[0,1,0,0],
		[0,0,1,0],
		[x,y,z,1],
		])

def mat_inverse(input1):
	input1 = input1.tolist()
	return np.matrix([
		[input1[0][0], input1[1][0], input1[2][0], 0, ],
		[input1[0][1], input1[1][1], input1[2][1], 0, ],
		[input1[0][2], input1[1][2], input1[2][2], 0, ],
		[-(input1[3][0] * input1[0][0] + input1[3][1] * input1[0][1] + input1[3][2] * input1[0][2]),
		 -(input1[3][0] * input1[1][0] + input1[3][1] * input1[1][1] + input1[3][2] * input1[1][2]), 
		 -(input1[3][0] * input1[2][0] + input1[3][1] * input1[2][1] + input1[3][2] * input1[2][2]), 1, ],
		])

def vec_dot_product(input1: vec3d, input2: vec3d) -> float:
	return input1.x * input2.x + input1.y * input2.y + input1.z * input2.z

def vec_length(input1: vec3d) -> float:
	return math.sqrt(vec_dot_product(input1, input1)) ## why was this sqrtf

def vec_normalise(input1: vec3d) -> vec3d:
	l = vec_length(input1)
	return vec_divide(input1, l)

def vec_to_np(vec, w=1):
	return np.array([vec.x, vec.y, vec.z, vec.w])

def np_to_vec(np):
	np=np.tolist()
	if np[3]:
		return vec3d(np[0], np[1], np[2], np[3])
	return vec3d(np[0], np[1], np[2])

def vec_normal(vec: vec3d):
		return math.sqrt(
			vec.x*vec.x+
			vec.y*vec.y+
			vec.z*vec.z)

def vec_cross_product(v1, v2):
		return vec3d(v1.y * v2.z - v1.z * v2.y,
			v1.z * v2.x - v1.x * v2.z,
			v1.x * v2.y - v1.y * v2.x)


class Window(Tk):
	def __init__(self):
		Tk.__init__(self)
		self.screen_width = self.winfo_screenwidth()
		self.screen_height = self.winfo_screenheight()
		self.title("Tkinter window")
		self.geometry("%dx%d" % (self.screen_width, self.screen_height))
		self.attributes('-topmost', True)
		# self.wm_attributes("-transparentcolor", "gray")
		self.canvas = Canvas(self, width=self.screen_width, height=self.screen_height)
		self.canvas.create_rectangle(0, 0, self.screen_width, self.screen_height, fill='gray', outline='gray')
		# self.wait_visibility()

		global vCamera
		vCamera = vec3d(0,0,-10)
		global fYaw
		fYaw = 0
		global fPitch
		fPitch = 0
		global fRoll
		fRoll = 0
		global light_direcion
		light_direcion = vec3d(0,-1,-1)
		self.timed_refresh()


	def draw(self, points, tag, shade):
		# print(f'Passed args: {points}, {tag}, {shade})')
		col = _from_rgb((
			int(200**shade),
			int(200**shade),
			int(200**shade)))
		xScale, yScale = 0.25, 0.5
		xTransform, yTransform = self.screen_width/3, self.screen_height/3
		self.canvas.create_polygon(
			points[0].x*xScale*self.screen_width+xTransform,
			points[0].y*yScale*self.screen_height+yTransform,
			points[1].x*xScale*self.screen_width+xTransform,		
			points[1].y*yScale*self.screen_height+yTransform,
			points[2].x*xScale*self.screen_width+xTransform,
			points[2].y*yScale*self.screen_height+yTransform,
			fill=col, outline='',tag=tag)
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

	def controls(self,
		forward='w',
		back='a',
		left='s',
		right='d',
		lookUp='i',
		lookDown='k',
		lookLeft='j',
		lookRight='l',):
		global vLookDir
		global fRoll
		def forward():
			global vCamera
			vForward = vec_multiply(vLookDir, 1)
			vCamera = vec_add(vCamera, vForward)
		def back():
			global vCamera
			vForward = vec_multiply(vLookDir, 1)
			vCamera = vec_subtract(vCamera, vForward)
		def left():
			global vCamera
			vForward = vec_multiply(vLookDir, 1)
			vRight = vec_cross_product(vUp, vForward)
			vCamera = vec_subtract(vCamera, vRight)
		def right():
			global vCamera
			vForward = vec_multiply(vLookDir, 1)
			vRight = vec_cross_product(vUp, vForward)
			vCamera = vec_add(vCamera, vRight)
		def lookUp():
			global fPitch
			fPitch -= 0.01
		def lookDown():
			global fPitch
			fPitch += 0.01
		def lookLeft():
			global fYaw
			fYaw -= 0.01
		def lookRight():
			global fYaw
			fYaw += 0.01
		action = {
		'forward' : forward(),
		'back' : back(),
		'left' : left(),
		'right' : right(),
		'lookUp' : lookUp(),
		'lookDown' : lookDown(),
		'lookLeft' : lookLeft(),
		'lookRight' : lookRight(),
		}
		return action

	def press(self, event):
		global vCamera
		global vLookDir
		global fYaw
		global fPitch
		global fRoll
		global vUp

		vForward = vec_multiply(vLookDir, 3)
		vRight = vec_cross_product(vUp, vForward)

		# print(f'vLookDir: {vLookDir.x}, y {vLookDir.y}, z: {vLookDir.z}, vForward: {vForward.x}, y {vForward.y}, z: {vForward.z}')


		# if event.char == 'w':
		# 	vCamera.z += 1
		# 	print(vCamera.z)
		# if event.char == 's':
		# 	vCamera.z -= 1

		if event.char == 'q':
			vCamera.y += 1
		if event.char == 'e':
			vCamera.y -= 1

		# if event.char == 'a':
		# 	vCamera.x += 1
		# if event.char == 'd':
		# 	vCamera.x -= 1

		if event.char == 'w':
			vCamera = vec_add(vCamera, vForward)
		if event.char == 's':
			vCamera = vec_subtract(vCamera, vForward)

		if event.char == 'a':
			vCamera = vec_subtract(vCamera, vRight)
		if event.char == 'd':
			vCamera = vec_add(vCamera, vRight)

		if event.char == 'i':
			fPitch += 0.05
		if event.char == 'k':
			fPitch -= 0.05

		if event.char == 'j':
			fYaw -= 0.05
		if event.char == 'l':
			fYaw += 0.05
		# print(f'{event.char}')

	def pipeline(self):
		#self.canvas.create_line(kwargs[0], kwargs[1], 300, 200, dash=(4, 2))
		#self.canvas.bind('<Motion>', self.mouse)
		self.mousex = self.winfo_pointerx()
		self.mousey = self.winfo_pointery()
		self.bind('<KeyPress>', self.press)
		# self.bind('<KeyRelease>', self.release)
		for obj in objectsGlobal:
			trisToRaster = []
			for corner in objectsGlobal[obj].triList:

				## World transformation
				triTransformed = triangle(
					multiply_vecmat(corner.p[0], world_matrix(objectsGlobal[obj].vCoords)),
					multiply_vecmat(corner.p[1], world_matrix(objectsGlobal[obj].vCoords)),
					multiply_vecmat(corner.p[2], world_matrix(objectsGlobal[obj].vCoords)))

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

				nClippedTriangles = Triangle_ClipAgainstPlane(vec3d(0,0,2.1), vec3d(0,0,1), triViewed, triangle(), triangle())
				for i in range(nClippedTriangles[0]):
					## Projection transformation
					triProjected = triangle(
						multiply_vecmat(nClippedTriangles[i+1].p[0], projection_matrix()),
						multiply_vecmat(nClippedTriangles[i+1].p[1], projection_matrix()),
						multiply_vecmat(nClippedTriangles[i+1].p[2], projection_matrix()),
						dp)

					triProjected.p[0] = vec_divide(triProjected.p[0], triProjected.p[0].w)
					triProjected.p[1] = vec_divide(triProjected.p[1], triProjected.p[1].w)
					triProjected.p[2] = vec_divide(triProjected.p[2], triProjected.p[2].w)

					trisToRaster.append(triProjected)

			trisToRaster.sort(reverse=True,key=getZs)

			trisFinal = []
			for triToRaster in trisToRaster:
				trisFinal.append(triToRaster)
				nNewTriangles = 1
				for i in range(4):
					while nNewTriangles > 0:
						test = trisFinal[0]
						trisFinal.pop(0)
						nNewTriangles -= 1
						if i == 0:
							nTrisToAdd = Triangle_ClipAgainstPlane(vec3d(0,0,0), vec3d(0,0,0), test, triangle(), triangle())
							# next
						elif i == 1:
							nTrisToAdd = Triangle_ClipAgainstPlane(vec3d(0,self.screen_height-1,0),vec3d(0,-1,0), test, triangle(), triangle())
							# next
						elif i == 2:
							nTrisToAdd = Triangle_ClipAgainstPlane(vec3d(0,0,0), vec3d(1,0,0), test, triangle(), triangle())
							# next
						elif i == 3:
							nTrisToAdd = Triangle_ClipAgainstPlane(vec3d(0,self.screen_width-1,0,0), vec3d(-1,0,0), test, triangle(), triangle())
							# next
						for w in range(nTrisToAdd[0]):
							trisFinal.append(nTrisToAdd[w+1])

			for tri in trisFinal:
				self.draw(tri.p, obj, tri.dp)


def world_matrix(trans):
	matTrans = mat_make_trans(trans[0],trans[1],trans[2],)
	# matTrans = mat_make_trans(0,10,5)
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

	matProj = np.matrix([
		[aspectRatio*fovRad,0,0,0],
		[0,fovRad,0,0],
		[0,0,znorm,1],
		[0,0,(-zfar*znear)/(zfar-znear),0]
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
		[math.cos(angle),-(math.sin(angle)),0,0],
		[math.sin(angle),math.cos(angle),0,0],
		[0,0,1,0],
		[0,0,0,1]
		])
	return matRotZ


def init_camera():
	global vCamera
	global vLookDir
	global vTarget
	global fYaw
	global fPitch
	global vUp

	vTarget = vec3d(0,0,1)
	vUp = vec3d(0,-1,0)
	# print(f'vTarget: {vTarget.x},{vTarget.y},{vTarget.z}')
	
	matCameraRotYaw = matRotY(fYaw) # Yaw pivots origin
	matCameraRotPitch = matRotX(fPitch)

	matCameraRot = mat_multiply(matCameraRotPitch, matCameraRotYaw)
	vLookDir = multiply_vecmat(vTarget, matCameraRot)

	vTarget = vec_add(vCamera, vLookDir)

	# print(vLookDir.x, vLookDir.y, vLookDir.z, )
	matCamera = mat_point_at(vCamera, vTarget, vUp)

	# matView = np.linalg.inv(matCamera) ## Crash here if something.z == 0
	matView = mat_inverse(matCamera)

	# print(f'vLookDir: {vLookDir.x},{vLookDir.y},{vLookDir.z}')
	# print(f'vUp: {vUp.x},{vUp.y},{vUp.z}, vTarget: {vTarget.x},{vTarget.y},{vTarget.z}')

	return matView

def mat_point_at(pos, target, up):
	newForward = vec_subtract(target, pos)
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

# def perlin_array(shape = (size*scaling_factor*2+1, size*scaling_factor*2+1),
# 			scale=10, octaves = 12,     #scale = 100
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
##    max_arr = np.max(arr)
##    min_arr = np.min(arr)
##    norm_me = lambda x: (x-min_arr)/(max_arr - min_arr)
##    norm_me = np.vectorize(norm_me)
##    arr = norm_me(arr)
    return arr


def main():
	global objectsGlobal
	objectsGlobal = {
		'ship': object('D:\\Users\\Koope\\Desktop\\ship.obj', 'ship', (0,0,0),),
		'axis': object('D:\\Users\\Koope\\Desktop\\axis.obj', 'axis', (0,10,5),),
		'cube': object('D:\\Users\\Koope\\Desktop\\cube.obj', 'cube', (3,5,0),1),
		# 'teapot2': object('D:\\Users\\Koope\\Desktop\\teapot2.obj', 'teapot2', (0,0,0),1),
		}

	App = Window()
	App.mainloop()

if __name__ == '__main__':
	start = datetime.datetime.now()
	main()
	end = datetime.datetime.now()
	secselapsed = end-start
	fps = ticker/secselapsed.total_seconds()
	print(f'fps: {int(fps)}')
