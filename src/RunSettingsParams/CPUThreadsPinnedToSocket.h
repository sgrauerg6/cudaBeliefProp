/*
 * CPUThreadsPinnedToSocket.h
 *
 *  Created on: Dec 2, 2024
 *      Author: scott
 * 
 *  Class for setting and retrieving setting of CPU threads pinned to socket
 */

#include <string>
#include <cstdlib>
#include "RunEval/RunData.h"
#include "RunSettingsConstsEnums.h"

//class to adjust and retrieve settings corresponding to CPU threads pinned to socket
class CPUThreadsPinnedToSocket {
public:
  //adjust setting to specify that CPU threads to be pinned to socket or not
  //if true, set CPU threads to be pinned to socket via OMP_PLACES and OMP_PROC_BIND envionmental variable settings
  //if false, set OMP_PLACES and OMP_PROC_BIND environment variables to be blank
  //TODO: currently commented out since it doesn't seem to have any effect
  void operator()(bool cpu_threads_pinned) const {
    /*if (cpu_threads_pinned) {
      int success = system("export OMP_PLACES=\"sockets\"");
      if (success == 0) {
        std::cout << "export OMP_PLACES=\"sockets\" success" << std::endl;
      }
      success = system("export OMP_PROC_BIND=true");
      if (success == 0) {
        std::cout << "export OMP_PROC_BIND=true success" << std::endl;
      }
    }
    else {
      int success = system("export OMP_PLACES=");
      if (success == 0) {
        std::cout << "export OMP_PLACES= success" << std::endl;
      }
      success = system("export OMP_PROC_BIND=");
      if (success == 0) {
        std::cout << "export OMP_PROC_BIND= success" << std::endl;
      }
    }*/
  }

  //retrieve environment variable values corresponding to CPU threads being pinned to socket and return
  //as RunData structure
  RunData SettingsAsRunData() const {
    RunData pinned_threads_settings;
    const std::string omp_places_setting = (std::getenv("OMP_PLACES") == nullptr) ? "" : std::getenv("OMP_PLACES");
    const std::string omp_proc_bind_setting = (std::getenv("OMP_PROC_BIND") == nullptr) ? "" : std::getenv("OMP_PROC_BIND");
    const bool cpu_threads_pinned = ((omp_places_setting == "sockets") && (omp_proc_bind_setting == "true"));
    pinned_threads_settings.AddDataWHeader(std::string(run_environment::kCPUThreadsPinnedHeader), cpu_threads_pinned);
    pinned_threads_settings.AddDataWHeader(std::string(run_environment::kOmpPlacesHeader), omp_places_setting);
    pinned_threads_settings.AddDataWHeader(std::string(run_environment::kOmpProcBindHeader), omp_proc_bind_setting);
    return pinned_threads_settings;
  }
};