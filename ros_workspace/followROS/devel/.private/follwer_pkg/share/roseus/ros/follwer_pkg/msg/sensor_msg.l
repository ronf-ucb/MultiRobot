;; Auto-generated. Do not edit!


(when (boundp 'follwer_pkg::sensor_msg)
  (if (not (find-package "FOLLWER_PKG"))
    (make-package "FOLLWER_PKG"))
  (shadow 'sensor_msg (find-package "FOLLWER_PKG")))
(unless (find-package "FOLLWER_PKG::SENSOR_MSG")
  (make-package "FOLLWER_PKG::SENSOR_MSG"))

(in-package "ROS")
;;//! \htmlinclude sensor_msg.msg.html


(defclass follwer_pkg::sensor_msg
  :super ros::object
  :slots (_leftSensor _rightSensor _middleSensor ))

(defmethod follwer_pkg::sensor_msg
  (:init
   (&key
    ((:leftSensor __leftSensor) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:rightSensor __rightSensor) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:middleSensor __middleSensor) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _leftSensor __leftSensor)
   (setq _rightSensor __rightSensor)
   (setq _middleSensor __middleSensor)
   self)
  (:leftSensor
   (&optional __leftSensor)
   (if __leftSensor (setq _leftSensor __leftSensor)) _leftSensor)
  (:rightSensor
   (&optional __rightSensor)
   (if __rightSensor (setq _rightSensor __rightSensor)) _rightSensor)
  (:middleSensor
   (&optional __middleSensor)
   (if __middleSensor (setq _middleSensor __middleSensor)) _middleSensor)
  (:serialization-length
   ()
   (+
    ;; float32[] _leftSensor
    (* 4    (length _leftSensor)) 4
    ;; float32[] _rightSensor
    (* 4    (length _rightSensor)) 4
    ;; float32[] _middleSensor
    (* 4    (length _middleSensor)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _leftSensor
     (write-long (length _leftSensor) s)
     (dotimes (i (length _leftSensor))
       (sys::poke (elt _leftSensor i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _rightSensor
     (write-long (length _rightSensor) s)
     (dotimes (i (length _rightSensor))
       (sys::poke (elt _rightSensor i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _middleSensor
     (write-long (length _middleSensor) s)
     (dotimes (i (length _middleSensor))
       (sys::poke (elt _middleSensor i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _leftSensor
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _leftSensor (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _leftSensor i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _rightSensor
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _rightSensor (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _rightSensor i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _middleSensor
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _middleSensor (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _middleSensor i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get follwer_pkg::sensor_msg :md5sum-) "c64c0bbfafe9bd0d7b532b4815be2a0d")
(setf (get follwer_pkg::sensor_msg :datatype-) "follwer_pkg/sensor_msg")
(setf (get follwer_pkg::sensor_msg :definition-)
      "float32[] leftSensor
float32[] rightSensor
float32[] middleSensor
")



(provide :follwer_pkg/sensor_msg "c64c0bbfafe9bd0d7b532b4815be2a0d")


