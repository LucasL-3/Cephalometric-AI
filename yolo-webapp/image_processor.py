from PIL import Image
import os
import torch
from ultralytics import YOLO
import sys
import math
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return f"{self.x},{self.y}"

class Vector:
    def __init__(self, pa, pb):
        self.x = int(pb.x) - int(pa.x)
        self.y = int(pb.y) - int(pa.y)
    def __str__(self):
        return f"{self.x},{self.y}"

class Angle:
    def __init__(self, va, vb):
        self.va = va
        self.vb = vb
    def theta(self):
        dot_product = self.va.x * self.vb.x + self.va.y * self.vb.y
        mag_a = math.hypot(self.va.x, self.va.y)
        mag_b = math.hypot(self.vb.x, self.vb.y)
        theta = math.degrees(math.acos(dot_product / (mag_a * mag_b)))
        return theta

class Distance:
    def __init__(self, pa, pb):
        self.x = (int(pb.x) - int(pa.x)) ** 2
        self.y = (int(pb.y) - int(pa.y)) ** 2
    def dist(self):
        return (self.x + self.y) ** 0.5

def checkArg():
    if len(sys.argv) != 2:
        print("Please provide a file")
        sys.exit(0)

def readFile(filename):
	points = []
	f = open(filename, "r")
	for line in f.readlines():
		line = line.strip(" \t\n\r")
		x = float(line.split(",")[0])
		y = float(line.split(",")[1])
		points.append(Point(int(x),int(y)))
	f.close()
	return points

def getCross(va, vb):
    return va.x * vb.y - va.y * vb.x

def getODI(pa, pb, pc, pd, pe, pf, pg, ph):
    va = Vector(pa, pb)
    vb = Vector(pc, pd)
    vc = Vector(pe, pf)
    vd = Vector(pg, ph)

    aa = Angle(va, vb).theta()
    ab = Angle(vc, vd).theta()
    cb = getCross(vc, vd)
    
    if cb < 0:
        ab = -ab
    return aa + ab

def getAPDI(pa, pb, pc, pd, pe, pf, pg, ph, pi, pj):
    va = Vector(pa, pb)
    vb = Vector(pc, pd)
    vc = Vector(pe, pf)
    vd = Vector(pg, ph)
    ve = Vector(pi, pj)

    aa = Angle(va, vb).theta()
    ab = Angle(vb, vc).theta()
    ac = Angle(vd, ve).theta()

    cb = getCross(vb, vc)
    cc = getCross(vd, ve)
    
    if cb > 0:
        ab = -ab
    if cc < 0:
        ac = -ac
    return aa + ab + ac

def writeFile(filename, points, ANBtype, SNBtype, SNAtype, ODItype, APDItype, FHItype, FMAtype, mwtype):
    with open(filename, "w") as f:
        for point in points:
            f.write(f"{point}\n")
        f.write(f"{ANBtype}\n")
        f.write(f"{SNBtype}\n")
        f.write(f"{SNAtype}\n")
        f.write(f"{ODItype}\n")
        f.write(f"{APDItype}\n")
        f.write(f"{FHItype}\n")
        f.write(f"{FMAtype}\n")
        f.write(f"{mwtype}\n")
        
def analysis(filename):
    points = readFile(filename)

    va = Vector(points[1], points[0])
    vb = Vector(points[1], points[5])
    vc = Vector(points[1], points[0])
    vd = Vector(points[1], points[4])
    print("1. ANB")
    ANBtype = ''
    ANB = Angle(vc, vd).theta() - Angle(va, vb).theta()
    print(ANB)
    if ANB < 3.2:
        ANBtype = '3'
    elif ANB > 5.7:
        ANBtype = '2'
    else:
        ANBtype = '1'
    ANBtype = ['ANB',ANBtype,  3.2, ANB, 5.7]
    print(ANBtype)
    print()
    
    va = Vector(points[1], points[0])
    vb = Vector(points[1], points[5])
    print("2. SNB")
    SNBtype = ''
    SNB = Angle(va, vb).theta()
    if SNB < 74.6:
        SNBtype = '2'
    elif SNB > 78.7:
        SNBtype = '3'
    else:
        SNBtype = '1'
    SNBtype = ['SNB',SNBtype,  74.6, SNB, 78.7]
    print(SNB)
    print(SNBtype)
    print()
    
    va = Vector(points[1], points[0])
    vb = Vector(points[1], points[4])
    print("3. SNA")
    SNAtype = ''
    SNA = Angle(va, vb).theta()
    if SNA < 79.4:
        SNAtype = '3'
    elif SNA > 83.2:
        SNAtype = '2'
    else:
        SNAtype = '1'
    SNAtype = ['SNA',SNAtype,  79.4, SNA, 83.2]
    print(SNA)
    print(SNAtype)
    print()

    print("4. ODI")
    ODItype = ''
    ODI = getODI(points[7], points[9], points[5], points[4], points[3], points[2], points[16], points[17])
    if ODI < 68.4:
        ODItype = '3'
    elif ODI > 80.5:
        ODItype = '2'
    else:
        ODItype = '1'
    ODItype = ['ODI',ODItype,  68.4, ODI, 80.5]
    print(ODI)
    print(ODItype)
    print()
    
    print("5. APDI")
    APDItype = ''
    APDI = getAPDI(points[2], points[3], points[1], points[6], points[4], points[5], points[3], points[2], points[16], points[17])
    if APDI < 77.6:
        APDItype = '2'
    elif APDI > 85.2:
        APDItype = '3'
    else:
        APDItype = '1'
    APDItype = ['APDI',APDItype,  77.6, APDI, 85.2]
    print(APDI)
    print(APDItype)
    print()
    
    pfh = Distance(points[0], points[9]).dist()
    afh = Distance(points[1], points[7]).dist()
    print("6. FHI")
    FHItype = ''
    FHI = pfh/afh
    print(pfh / afh)
    if pfh / afh < 0.65:
        FHItype = '3'
    elif pfh / afh > 0.75:
        FHItype = '2'
    else:
        FHItype = '1'
    FHItype = ['FHI',FHItype,  0.65, FHI, 0.75]
    print(FHItype)
    print()
    
    va = Vector(points[0], points[1])
    vb = Vector(points[9], points[8])
    print("7. FMA")
    FMAtype = ''
    angle_fma = Angle(va, vb).theta()
    if angle_fma < 26.8:
        FMAtype = '3'
    elif angle_fma > 31.4:
        FMAtype = '2'
    else:
        FMAtype = '1'
    FMAtype = ['FMA',FMAtype,  26.8, angle_fma, 31.4]
    print(angle_fma)
    print(FMAtype)
    print()
    
    mw = Distance(points[10], points[11]).dist() / 10
    mwtype = ''
    if points[11].x < points[10].x:
        mw = -mw
    if mw >= 2:
        if mw <= 4.5:
            mwtype = '1'
        else:
            mwtype = '4'
    elif mw == 0:
        mwtype = '2'
    else:
        mwtype = '3'
    mwtype = ['MW',mwtype,  2.0, mw, 4.5]
    print("8. MW")
    print(mw)
    print(mwtype)
    return ANBtype, SNBtype, SNAtype, ODItype, APDItype, FHItype, FMAtype, mwtype
    #writeFile(filename, points, ANBtype, SNBtype, SNAtype, ODItype, APDItype, FHItype, FMAtype, mwtype)






# Load a model
model = YOLO("/Users/Lucas/Desktop/Cephalometric-AI/yolov8/runs/pose/train23/weights/best.pt")  # load a custom model
def yolo_inference(image_path, output_path):
    results = model(image_path)  # predict on an image
    print(results)


    for result in results:
        '''boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen'''
        result.save(filename=output_path)  
        print(result.keypoints)
        array = result.keypoints.xyn.cpu().numpy()
        filename = "./static/uploads/xy.txt"
        np.savetxt(filename, array.reshape(1, -1), fmt='%6.3f', delimiter=' ')
        
        tensor_to_points_file(result.keypoints.xy)
        ANBtype, SNBtype, SNAtype, ODItype, APDItype, FHItype, FMAtype, mwtype=analysis("points.txt")

    return ANBtype, SNBtype, SNAtype, ODItype, APDItype, FHItype, FMAtype, mwtype
def tensor_to_points_file(tensor, filename="points.txt"):
    # Open the file in write mode
    with open(filename, "w") as file:
        # Iterate through the tensor and write each point to the file
        for point in tensor[0]:
            x, y = point
            file.write(f"{x.item()},{y.item()}\n")

'''def rotate_and_resize_image(image_path, output_path, angle=290, size=(350, 350)):
    try:
        with Image.open(image_path) as img:
            rotated_img = img.rotate(-angle, expand=True)  # Rotate by the specified angle
            resized_img = rotated_img.resize(size)  # Resize the image
            resized_img.save(output_path)
        print(f"Image rotated by {angle} degrees, resized to {size}, and saved to: {output_path}")
        print(output_path)
    except Exception as e:
        print(f"Error rotating and resizing image: {str(e)}")'''
