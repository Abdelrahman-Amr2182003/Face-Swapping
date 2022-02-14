
import dlib
import numpy as np
import cv2
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_land_marks(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask=np.zeros_like(gray)
    faces=detector(gray)
    my_ret=None
    for face in faces:
        land_marks=predictor(gray,face)
        land_marks_points=[]
        for n in range(68):
            x=land_marks.part(n).x
            y=land_marks.part(n).y
            land_marks_points.append([x,y])
            #cv2.circle(img,(x,y),5,(255,0,0),-1)
        pts=np.array(land_marks_points,dtype='int32')
        hull=cv2.convexHull(pts)
        #cv2.polylines(img,[hull],True,(0,255,0),2)
        cv2.fillConvexPoly(mask,hull,255)
        res=cv2.bitwise_and(img,img,mask=mask)
        rect=cv2.boundingRect(hull)
        my_ret= land_marks_points,mask,res,rect
    return my_ret
def extract_index(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index
def swap(img,target):
    res_face=np.zeros_like(target)
    my_ret=get_land_marks(img)
    my_ret2=get_land_marks(target)
    if my_ret is not None:
        land_marks_points_s,mask_s,res_s,rect_s=my_ret
        pts_arr=np.array(land_marks_points_s)
        subdiv=cv2.Subdiv2D(rect_s)
        subdiv.insert(land_marks_points_s)
        triangels=subdiv.getTriangleList()
        trs_indexes=[]
        for t in triangels:
            pt1=(t[0],t[1])
            pt2=(t[2],t[3])
            pt3=(t[4],t[5])
            #cv2.line(img,pt1,pt2,(255,0,0),3)
            #cv2.line(img,pt1,pt3,(255,0,0),3)
            #cv2.line(img,pt2,pt3,(255,0,0),3)
            pt1_index=np.where((pts_arr==pt1).all(axis=1))
            pt2_index=np.where((pts_arr==pt2).all(axis=1))
            pt3_index=np.where((pts_arr==pt3).all(axis=1))
            index_1=extract_index(pt1_index)
            index_2=extract_index(pt2_index)
            index_3=extract_index(pt3_index)
            trs_indexes.append([index_1,index_2,index_3])
    if my_ret2 is not None and my_ret is not None:
        land_marks_points_t,mask_tr,res_t,rect_tr=my_ret2
        for tr in trs_indexes:
            pt1_src=tuple(land_marks_points_s[tr[0]])
            pt2_src=tuple(land_marks_points_s[tr[1]])
            pt3_src=tuple(land_marks_points_s[tr[2]])
            
            pt1_t=tuple(land_marks_points_t[tr[0]])
            pt2_t=tuple(land_marks_points_t[tr[1]])
            pt3_t=tuple(land_marks_points_t[tr[2]])
            
            pts_src=np.array([pt1_src,pt2_src,pt3_src])
            pts_t=np.array([pt1_t,pt2_t,pt3_t])
            
            rect_tr_s=cv2.boundingRect(pts_src)
            rect_tr_t=cv2.boundingRect(pts_t)
            
            x_s,y_s,w_s,h_s=rect_tr_s
            x_t,y_t,w_t,h_t=rect_tr_t
            
            roi_s=img[y_s:y_s+h_s,x_s:x_s+w_s]
            roi_t=target[y_t:y_t+h_t,x_t:x_t+w_t]
            new_pts_src=np.array([(pt1_src[0]-x_s,pt1_src[1]-y_s),
                                  (pt2_src[0]-x_s,pt2_src[1]-y_s),
                                  (pt3_src[0]-x_s,pt3_src[1]-y_s)
                                 ],dtype='int32')
            new_pts_t=np.array([(pt1_t[0]-x_t,pt1_t[1]-y_t),
                                  (pt2_t[0]-x_t,pt2_t[1]-y_t),
                                  (pt3_t[0]-x_t,pt3_t[1]-y_t)
                                 ],dtype='int32')
            
            mask_s=np.zeros_like(cv2.cvtColor(roi_s,cv2.COLOR_BGR2GRAY))
            mask_t=np.zeros_like(cv2.cvtColor(roi_t,cv2.COLOR_BGR2GRAY))
            
            hull_s=cv2.convexHull(new_pts_src)
            hull_t=cv2.convexHull(new_pts_t)
            cv2.fillConvexPoly(mask_s,hull_s,255)
            cv2.fillConvexPoly(mask_t,hull_t,255)
            roi_s=cv2.bitwise_and(roi_s,roi_s,mask=mask_s)
            roi_t=cv2.bitwise_and(roi_t,roi_t,mask=mask_t)
            matrix=cv2.getAffineTransform(np.float32(new_pts_src),np.float32(new_pts_t))
            alligned_tr=cv2.warpAffine(roi_s,matrix,(w_t,h_t))
            res_face[y_t:y_t+h_t,x_t:x_t+w_t]=cv2.add(res_face[y_t:y_t+h_t,x_t:x_t+w_t],alligned_tr)
            _,mask_new_face=cv2.threshold(res_face,1,255,cv2.THRESH_BINARY)
        mask_new_face_inv=cv2.bitwise_not(mask_tr)
        #print(mask_new_face.shape)
        #print(target.shape)
        #print(mask_new_face_inv.shape)
        target_no_face=cv2.bitwise_and(target,target,mask=mask_new_face_inv)
        total=cv2.add(target_no_face,res_face)
        (x, y, w, h) = rect_tr
        center_face = (int((x +x + w) / 2), int((y + y + h) / 2))
        seamlessclone = cv2.seamlessClone(total, target, mask_new_face, center_face,cv2.NORMAL_CLONE)
        return seamlessclone
            
cap=cv2.VideoCapture(0)
ret,frame=cap.read()
h,w,c=frame.shape
result = cv2.VideoWriter('swapping.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         30, (700,700))
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(700,700))
    #frame=cv2.flip(frame,1,-1)
    my_ret=swap(cv2.imread('bradley_cooper.jpg'),frame)
    if my_ret is not None:
        final=my_ret
        
        #cv2.imshow('total',total)
        cv2.imshow('final',final)
        result.write(final)
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    if key &0xFF ==ord('q'):
        break
cv2.destroyAllWindows()