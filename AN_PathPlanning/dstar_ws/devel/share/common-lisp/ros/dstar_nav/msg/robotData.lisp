; Auto-generated. Do not edit!


(cl:in-package dstar_nav-msg)


;//! \htmlinclude robotData.msg.html

(cl:defclass <robotData> (roslisp-msg-protocol:ros-message)
  ((robPos
    :reader robPos
    :initarg :robPos
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (goalPos
    :reader goalPos
    :initarg :goalPos
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (robOrient
    :reader robOrient
    :initarg :robOrient
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (proxVec
    :reader proxVec
    :initarg :proxVec
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (sense3D
    :reader sense3D
    :initarg :sense3D
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (proxDist
    :reader proxDist
    :initarg :proxDist
    :type cl:float
    :initform 0.0))
)

(cl:defclass robotData (<robotData>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <robotData>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'robotData)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dstar_nav-msg:<robotData> is deprecated: use dstar_nav-msg:robotData instead.")))

(cl:ensure-generic-function 'robPos-val :lambda-list '(m))
(cl:defmethod robPos-val ((m <robotData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dstar_nav-msg:robPos-val is deprecated.  Use dstar_nav-msg:robPos instead.")
  (robPos m))

(cl:ensure-generic-function 'goalPos-val :lambda-list '(m))
(cl:defmethod goalPos-val ((m <robotData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dstar_nav-msg:goalPos-val is deprecated.  Use dstar_nav-msg:goalPos instead.")
  (goalPos m))

(cl:ensure-generic-function 'robOrient-val :lambda-list '(m))
(cl:defmethod robOrient-val ((m <robotData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dstar_nav-msg:robOrient-val is deprecated.  Use dstar_nav-msg:robOrient instead.")
  (robOrient m))

(cl:ensure-generic-function 'proxVec-val :lambda-list '(m))
(cl:defmethod proxVec-val ((m <robotData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dstar_nav-msg:proxVec-val is deprecated.  Use dstar_nav-msg:proxVec instead.")
  (proxVec m))

(cl:ensure-generic-function 'sense3D-val :lambda-list '(m))
(cl:defmethod sense3D-val ((m <robotData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dstar_nav-msg:sense3D-val is deprecated.  Use dstar_nav-msg:sense3D instead.")
  (sense3D m))

(cl:ensure-generic-function 'proxDist-val :lambda-list '(m))
(cl:defmethod proxDist-val ((m <robotData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dstar_nav-msg:proxDist-val is deprecated.  Use dstar_nav-msg:proxDist instead.")
  (proxDist m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <robotData>) ostream)
  "Serializes a message object of type '<robotData>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'robPos))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'robPos))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'goalPos))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'goalPos))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'robOrient))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'robOrient))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'proxVec))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'proxVec))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'sense3D))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'sense3D))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'proxDist))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <robotData>) istream)
  "Deserializes a message object of type '<robotData>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'robPos) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'robPos)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'goalPos) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'goalPos)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'robOrient) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'robOrient)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'proxVec) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'proxVec)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'sense3D) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'sense3D)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'proxDist) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<robotData>)))
  "Returns string type for a message object of type '<robotData>"
  "dstar_nav/robotData")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'robotData)))
  "Returns string type for a message object of type 'robotData"
  "dstar_nav/robotData")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<robotData>)))
  "Returns md5sum for a message object of type '<robotData>"
  "e646730a5027be23477dd7883ffccb5d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'robotData)))
  "Returns md5sum for a message object of type 'robotData"
  "e646730a5027be23477dd7883ffccb5d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<robotData>)))
  "Returns full string definition for message of type '<robotData>"
  (cl:format cl:nil "float32[] robPos~%float32[] goalPos~%float32[] robOrient~%float32[] proxVec~%float32[] sense3D~%float32 proxDist~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'robotData)))
  "Returns full string definition for message of type 'robotData"
  (cl:format cl:nil "float32[] robPos~%float32[] goalPos~%float32[] robOrient~%float32[] proxVec~%float32[] sense3D~%float32 proxDist~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <robotData>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'robPos) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'goalPos) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'robOrient) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'proxVec) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'sense3D) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <robotData>))
  "Converts a ROS message object to a list"
  (cl:list 'robotData
    (cl:cons ':robPos (robPos msg))
    (cl:cons ':goalPos (goalPos msg))
    (cl:cons ':robOrient (robOrient msg))
    (cl:cons ':proxVec (proxVec msg))
    (cl:cons ':sense3D (sense3D msg))
    (cl:cons ':proxDist (proxDist msg))
))
