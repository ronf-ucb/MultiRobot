// Generated by gencpp from file dstar_nav/edgeRequest.msg
// DO NOT EDIT!


#ifndef DSTAR_NAV_MESSAGE_EDGEREQUEST_H
#define DSTAR_NAV_MESSAGE_EDGEREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace dstar_nav
{
template <class ContainerAllocator>
struct edgeRequest_
{
  typedef edgeRequest_<ContainerAllocator> Type;

  edgeRequest_()
    : point1()
    , point2()  {
    }
  edgeRequest_(const ContainerAllocator& _alloc)
    : point1(_alloc)
    , point2(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _point1_type;
  _point1_type point1;

   typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _point2_type;
  _point2_type point2;





  typedef boost::shared_ptr< ::dstar_nav::edgeRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::dstar_nav::edgeRequest_<ContainerAllocator> const> ConstPtr;

}; // struct edgeRequest_

typedef ::dstar_nav::edgeRequest_<std::allocator<void> > edgeRequest;

typedef boost::shared_ptr< ::dstar_nav::edgeRequest > edgeRequestPtr;
typedef boost::shared_ptr< ::dstar_nav::edgeRequest const> edgeRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::dstar_nav::edgeRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::dstar_nav::edgeRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace dstar_nav

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'dstar_nav': ['/home/austinnguyen517/Documents/Research/BML/MultiRobot/AN_PathPlanning/dstar_ws/src/dstar_nav/msg'], 'std_msgs': ['/opt/ros/melodic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::dstar_nav::edgeRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dstar_nav::edgeRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dstar_nav::edgeRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dstar_nav::edgeRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dstar_nav::edgeRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dstar_nav::edgeRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::dstar_nav::edgeRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d21ddc5b335d396229c93135240e91ac";
  }

  static const char* value(const ::dstar_nav::edgeRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd21ddc5b335d3962ULL;
  static const uint64_t static_value2 = 0x29c93135240e91acULL;
};

template<class ContainerAllocator>
struct DataType< ::dstar_nav::edgeRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "dstar_nav/edgeRequest";
  }

  static const char* value(const ::dstar_nav::edgeRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::dstar_nav::edgeRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float32[] point1\n"
"float32[] point2\n"
;
  }

  static const char* value(const ::dstar_nav::edgeRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::dstar_nav::edgeRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.point1);
      stream.next(m.point2);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct edgeRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::dstar_nav::edgeRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::dstar_nav::edgeRequest_<ContainerAllocator>& v)
  {
    s << indent << "point1[]" << std::endl;
    for (size_t i = 0; i < v.point1.size(); ++i)
    {
      s << indent << "  point1[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.point1[i]);
    }
    s << indent << "point2[]" << std::endl;
    for (size_t i = 0; i < v.point2.size(); ++i)
    {
      s << indent << "  point2[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.point2[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // DSTAR_NAV_MESSAGE_EDGEREQUEST_H
