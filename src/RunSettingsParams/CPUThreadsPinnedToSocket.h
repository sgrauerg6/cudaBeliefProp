/*
Copyright (C) 2024 Scott Grauer-Gray

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/**
 * @file CPUThreadsPinnedToSocket.h
 * @author Scott Grauer-Gray
 * @brief Class for setting and retrieving setting of CPU threads pinned to socket
 * 
 * @copyright Copyright (c) 2024
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