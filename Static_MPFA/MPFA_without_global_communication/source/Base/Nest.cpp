#include "Nest.h"

/*****
 * The iAnt nest needs to keep track of four things:
 *
 * [1] location
 * [2] nest id 
 * [3] site fidelity
 * [4] pheromone trails
 *
 *****/
	Nest::Nest(){}
	Nest::Nest(CVector2   location)
{
    /* required initializations */
	   nestLocation    = location;
	   departLocation  = location;
    PheromoneList.clear();
    FidelityList.clear();
    DensityOnFidelity.clear(); //qilu 09/11/2016
    FoodList.clear(); //qilu 09/07/2016
    NewLocation = location; //qilu 09/19/2016
    num_collected_tags=0;
    visited_time_point_in_minute=0;
    nest_idx=-1;
    travel_flag =0;
}

/*****
 *****/

/*****
 * Return the nest's location.
 *****/
CVector2 Nest::GetLocation() {
    return nestLocation;
}

CVector2 Nest::GetDepartLocation() {
    return departLocation;
}
void Nest::SetLocation() {
    nestLocation=CVector2(0.0, 0.0);
}

void Nest::SetLocation(CVector2 newLocation) {
    nestLocation = newLocation;
}

void Nest::SetDepartLocation(CVector2 location){
	departLocation = location;
	}
void Nest:: SetNestIdx(size_t idx){
     nest_idx = idx;
 }
 
size_t Nest:: GetNestIdx(){
     return nest_idx;
 } 

size_t Nest:: GetCollectedTagNum(){
     return num_collected_tags;
 } 
 
int Nest:: GetTravelFlag(){
	return travel_flag;
	}

void Nest:: SetTravelFlag(int flag){
	travel_flag = flag;
	}
void Nest::UpdateNestLocation(){ //qilu 09/10/2016
    CVector2 Sum_locations = CVector2(0.0, 0.0);
    CVector2 placementPosition;
    size_t Num_points = 0;
    CVector2 offset;

      if(num_collected_tags >= 1){ //if full loaded, then travel to the center
        NewLocation = CVector2(0.0, 0.0);
        departLocation = nestLocation;
        travel_flag = 1;
	 }
}
        
