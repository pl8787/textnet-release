#ifndef TEXTNET_STATISTIC_STATISTIC_H_
#define TEXTNET_STATISTIC_STATISTIC_H_

#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include "../global.h"

#include "zmq.h"
#include "pthread.h"
#include "unistd.h"

#include "../net/net.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../io/json/json.h"

/*! \brief namespace of textnet */
namespace textnet {
/*! \brief namespace of net defintiion */
namespace statistic {
  
using namespace std;
using namespace layer;
using namespace net;
using namespace mshadow;

typedef int MsgType;
const int NONE_INFO = 0;
const int PARAM_INFO = 1;
const int NODE_INFO = 2;
const int STATE_INFO = 3;
 
class Statistic {
 public:
 
  Statistic(INet* net_):net(net_) {

  }

  virtual ~Statistic(void) {

  }

  string RecieveMsg() {
    zmq_msg_t request;
    zmq_msg_init(&request);
    zmq_msg_recv(&request, responder, 0);
    int size = zmq_msg_size (&request);
    string msg_rev((char *)(zmq_msg_data(&request)), size);
    zmq_msg_close (&request);
    utils::Printf("Received: %s\n",msg_rev.c_str());
	return msg_rev;
  }
  
  bool SendMsg(string msg) {
    zmq_msg_t reply;
    zmq_msg_init_size(&reply, msg.size());
    memcpy(zmq_msg_data(&reply), msg.c_str(), msg.size());
    zmq_msg_send(&reply, responder, 0);
    zmq_msg_close(&reply);
	return true;
  }

  void Router(string msg) {
    Json::Value msg_root;
    Json::Value res_root;
    string res_msg;
    if (reader.parse(msg, msg_root)) {
      MsgType msg_type = msg_root["msg_type"].asInt();
      switch (msg_type) {
        case NONE_INFO:
          res_root = NoneMsg(msg_root);
          break;
        case PARAM_INFO:
          res_root = ParamMsg(msg_root);
          break;
        case NODE_INFO:
          res_root = NodeMsg(msg_root);
          break;
        case STATE_INFO:
          res_root = StateMsg(msg_root);
          break;
        default:
          utils::Printf("Unknow type of message.\n");
          break;
      }
      res_msg = writer.write(res_root);
      if (SendMsg(res_msg)) {
        utils::Printf("Send msg ok!\n");
      } else {
        utils::Printf("Send msg error!\n");
      }
    } else {
      utils::Printf("Can not parse to json.");
    }
  }

  Json::Value NoneMsg(Json::Value &req_root) {
    Json::Value res_root;
    res_root["data"] = "Hello World!";
    return res_root;
  }
  
  Json::Value ParamMsg(Json::Value &req_root) {
	Json::Value res_root = net->StatisticParam(req_root);
    return res_root;
  }

  Json::Value NodeMsg(Json::Value &req_root) {
	Json::Value res_root = net->StatisticNode(req_root);
    return res_root; 
  }

  Json::Value StateMsg(Json::Value &req_root) {
	Json::Value res_root = net->StatisticState(req_root);
    return res_root;
  }

  static void * Listening(void * data) {
	Statistic * pstatic = (Statistic *)data;
    while (true) {
      utils::Printf("I am listening...\n");
      string req_msg = pstatic->RecieveMsg();
      pstatic->Router(req_msg);
	}
	return NULL;
  }

  void Start() {
	utils::Check(net!=NULL, "Net is not avaliable.");
    context = zmq_init(1);
	responder = zmq_socket(context, ZMQ_REP);
	zmq_bind(responder, "tcp://127.0.0.1:5000");
	utils::Printf("Binding on port 5000.");
    pthread_create(&thread_listen, NULL, Listening, (void *)this);
  }
 
  void Stop() {
	pthread_join(thread_listen, &retval);
  }

 protected:
  // Net need monitor 
  INet * net;
  // zmq context
  void * context;
  // zmq responder
  void * responder;
  // Thread for server listening
  pthread_t thread_listen;
  // Thread return value
  void *retval;
  // Json Reader
  Json::Reader reader;
  // Json Writer
  Json::FastWriter writer;
};

}  // namespace statistic
}  // namespace textnet
#endif  // TEXTNET_NET_NET_H_
