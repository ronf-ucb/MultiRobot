; Auto-generated. Do not edit!


(cl:in-package follwer_pkg-msg)


;//! \htmlinclude sensor_msg.msg.html

(cl:defclass <sensor_msg> (roslisp-msg-protocol:ros-message)
  ((leftSensor
    :reader leftSensor
    :initarg :leftSensor
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (rightSensor
    :reader rightSensor
    :initarg :rightSensor
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (middleSensor
    :reader middleSensor
    :initarg :middleSensor
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass sensor_msg (<sensor_msg>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <sensor_msg>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'sensor_msg)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name follwer_pkg-msg:<sensor_msg> is deprecated: use follwer_pkg-msg:sensor_msg instead.")))

(cl:ensure-generic-function 'leftSensor-val :lambda-list '(m))
(cl:defmethod leftSensor-val ((m <sensor_msg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader follwer_pkg-msg:leftSensor-val is deprecated.  Use follwer_pkg-msg:leftSensor instead.")
  (leftSensor m))

(cl:ensure-generic-function 'rightSensor-val :lambda-list '(m))
(cl:defmethod rightSensor-val ((m <sensor_msg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader follwer_pkg-msg:rightSensor-val is deprecated.  Use follwer_pkg-msg:rightSensor instead.")
  (rightSensor m))

(cl:ensure-generic-function 'middleSensor-val :lambda-list '(m))
(cl:defmethod middleSensor-val ((m <sensor_msg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader follwer_pkg-msg:middleSensor-val is deprecated.  Use follwer_pkg-msg:middleSensor instead.")
  (middleSensor m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <sensor_msg>) ostream)
  "Serializes a message object of type '<sensor_msg>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'leftSensor))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'leftSensor))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'rightSensor))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'rightSensor))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'middleSensor))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'middleSensor))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <sensor_msg>) istream)
  "Deserializes a message object of type '<sensor_msg>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'leftSensor) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'leftSensor)))
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
  (cl:setf (cl:slot-value msg 'rightSensor) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'rightSensor)))
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
  (cl:setf (cl:slot-value msg 'middleSensor) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'middleSensor)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<sensor_msg>)))
  "Returns string type for a message object of type '<sensor_msg>"
  "follwer_pkg/sensor_msg")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'sensor_msg)))
  "Returns string type for a message object of type 'sensor_msg"
  "follwer_pkg/sensor_msg")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<sensor_msg>)))
  "Returns md5sum for a message object of type '<sensor_msg>"
  "c64c0bbfafe9bd0d7b532b4815be2a0d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'sensor_msg)))
  "Returns md5sum for a message object of type 'sensor_msg"
  "c64c0bbfafe9bd0d7b532b4815be2a0d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<sensor_msg>)))
  "Returns full string definition for message of type '<sensor_msg>"
  (cl:format cl:nil "float32[] leftSensor~%float32[] rightSensor~%float32[] middleSensor~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'sensor_msg)))
  "Returns full string definition for message of type 'sensor_msg"
  (cl:format cl:nil "float32[] leftSensor~%float32[] rightSensor~%float32[] middleSensor~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <sensor_msg>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'leftSensor) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'rightSensor) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'middleSensor) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <sensor_msg>))
  "Converts a ROS message object to a list"
  (cl:list 'sensor_msg
    (cl:cons ':leftSensor (leftSensor msg))
    (cl:cons ':rightSensor (rightSensor msg))
    (cl:cons ':middleSensor (middleSensor msg))
))
