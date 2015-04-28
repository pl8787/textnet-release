#ifndef TEXTNET_STATISTIC_STATISTIC_H_
#define TEXTNET_STATISTIC_STATISTIC_H_

#pragma once

#if REALTIME_SERVER==1

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
 
  Statistic(void) {
    url_port = "tcp://127.0.0.1:5000";
	net = NULL;
  }

  virtual ~Statistic(void) {

  }

  void SetUrlPort(string url_port_) {
    url_port = url_port_;
  }

  void SetNet(INet* net_) {
    net = net_;
  }

  string RecieveMsg() {
	int rc = 0;
    zmq_msg_t request;
    rc = zmq_msg_init(&request);
	utils::Check(rc == 0, "In RecieveMsg, zmq_msg_init.");
    rc = zmq_msg_recv(&request, responder, 0);
	utils::Check(rc != -1, "In RecieveMsg, zmq_msg_recv.");

    int size = zmq_msg_size (&request);
    string msg_rev((char *)(zmq_msg_data(&request)), size);
    zmq_msg_close (&request);

	// char client_name[256];
	// size_t name_size = 256;
    // rc = zmq_getsockopt(responder, ZMQ_LAST_ENDPOINT, client_name, &name_size);
	// utils::Check(rc == 0, "In RecieveMsg, zmq_getsockopt.");
    utils::Printf("[Server] Received: %d\n", rc); //client_name);
	return msg_rev;
  }
  
  bool SendMsg(string msg) {
	int rc = 0;
    zmq_msg_t reply;
    rc = zmq_msg_init_size(&reply, msg.size());
	utils::Check(rc == 0, "In RecieveMsg, zmq_msg_init_size");

    memcpy(zmq_msg_data(&reply), msg.c_str(), msg.size());

    rc = zmq_msg_send(&reply, responder, 0);
	utils::Check(rc != -1, "In RecieveMsg, zmq_getsockopt.");
    utils::Printf("[Server] Send msg: %d\n", rc);

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
      if (!SendMsg(res_msg)) {
        utils::Printf("[Server] Send msg error!\n");
      }
    } else {
      utils::Printf("[Server] Can not parse to json.\n");
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
    utils::Printf("[Server] I am listening ...\n");
    while (true) {
      string req_msg = pstatic->RecieveMsg();
      pstatic->Router(req_msg);
	}
	return NULL;
  }

  void Start() {
	utils::Check(net!=NULL, "Net is not avaliable.");
    context = zmq_init(1);
	responder = zmq_socket(context, ZMQ_REP);
	zmq_bind(responder, url_port.c_str());
	utils::Printf("[Server] Binding on %s.\n", url_port.c_str());
    pthread_create(&thread_listen, NULL, Listening, (void *)this);
  }
 
  void Stop() {
	pthread_join(thread_listen, &retval);
	zmq_close (responder);
	zmq_term (context);
  }

 protected:
  // Server url and port
  string url_port;
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

#endif // REALTIME_SERVER

#endif  // TEXTNET_NET_NET_H_
