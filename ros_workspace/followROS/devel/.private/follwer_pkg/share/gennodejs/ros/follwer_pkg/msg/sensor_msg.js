// Auto-generated. Do not edit!

// (in-package follwer_pkg.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class sensor_msg {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.leftSensor = null;
      this.rightSensor = null;
      this.middleSensor = null;
    }
    else {
      if (initObj.hasOwnProperty('leftSensor')) {
        this.leftSensor = initObj.leftSensor
      }
      else {
        this.leftSensor = [];
      }
      if (initObj.hasOwnProperty('rightSensor')) {
        this.rightSensor = initObj.rightSensor
      }
      else {
        this.rightSensor = [];
      }
      if (initObj.hasOwnProperty('middleSensor')) {
        this.middleSensor = initObj.middleSensor
      }
      else {
        this.middleSensor = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type sensor_msg
    // Serialize message field [leftSensor]
    bufferOffset = _arraySerializer.float32(obj.leftSensor, buffer, bufferOffset, null);
    // Serialize message field [rightSensor]
    bufferOffset = _arraySerializer.float32(obj.rightSensor, buffer, bufferOffset, null);
    // Serialize message field [middleSensor]
    bufferOffset = _arraySerializer.float32(obj.middleSensor, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type sensor_msg
    let len;
    let data = new sensor_msg(null);
    // Deserialize message field [leftSensor]
    data.leftSensor = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [rightSensor]
    data.rightSensor = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [middleSensor]
    data.middleSensor = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.leftSensor.length;
    length += 4 * object.rightSensor.length;
    length += 4 * object.middleSensor.length;
    return length + 12;
  }

  static datatype() {
    // Returns string type for a message object
    return 'follwer_pkg/sensor_msg';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'c64c0bbfafe9bd0d7b532b4815be2a0d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32[] leftSensor
    float32[] rightSensor
    float32[] middleSensor
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new sensor_msg(null);
    if (msg.leftSensor !== undefined) {
      resolved.leftSensor = msg.leftSensor;
    }
    else {
      resolved.leftSensor = []
    }

    if (msg.rightSensor !== undefined) {
      resolved.rightSensor = msg.rightSensor;
    }
    else {
      resolved.rightSensor = []
    }

    if (msg.middleSensor !== undefined) {
      resolved.middleSensor = msg.middleSensor;
    }
    else {
      resolved.middleSensor = []
    }

    return resolved;
    }
};

module.exports = sensor_msg;
