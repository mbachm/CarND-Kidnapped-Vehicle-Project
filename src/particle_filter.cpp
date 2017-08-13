/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  //set number of particles
  num_particles = 1000;
  
  //gaussian/normal distributions
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  //new number generator for each run
  default_random_engine gen;
  
  //Initalize particles for complete list
  for(int i = 0; i< num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
    weights.push_back(1.0);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  
  //noise
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];
  
  //new number generator for each run
  default_random_engine gen;
  
  for(int i = 0; i < num_particles; ++i) {
    //Uncomment to debug
    //cout << "particle " << i << " (" << particles[i].x << ", " << particles[i].y << ", " << particles[i].theta << ")" << endl;
    
    //safe theta just for safety
    const double theta = particles[i].theta;
    
    //check if yaw_rate is zero (or too near zero)
    //like discussed in motion model lesson 12
    if(fabs(yaw_rate) > 0.0001) {
      const double c0 = velocity / yaw_rate;
      const double c1 = yaw_rate * delta_t;
      particles[i].x += c0 * (sin(theta + c1) - sin(theta));
      particles[i].y += c0 * (cos(theta)- cos(theta + c1));
      particles[i].theta += c1;
    }
    else {
      const double distance_in_time = velocity * delta_t;
      particles[i].x += distance_in_time * cos(yaw_rate);
      particles[i].y += distance_in_time * sin(yaw_rate);
    }
    
    normal_distribution<double> dist_x(particles[i].x, std_x);
    normal_distribution<double> dist_y(particles[i].y, std_y);
    normal_distribution<double> dist_theta(particles[i].theta, std_theta);
    
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    
    //Uncomment to debug
    //cout << "particle " << i << " (" << particles[i].x << ", " << particles[i].y << ", " << particles[i].theta << ")" << endl;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for(int i = 0; i < observations.size(); ++i) {
    double dist = numeric_limits<double>::max();
    int nearest_id = 0;
    
    for(LandmarkObs pre: predicted) {
      const int distancex = pow((pre.x - observations[i].x), 2);
      const int distancey = pow((pre.y - observations[i].y), 2);
      
      const double calcdistance = sqrt(distancex + distancey);
      if(calcdistance < dist) {
        nearest_id = pre.id;
        dist = calcdistance;
      }
    }
    observations[i].id = nearest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  double totalWeight = 0.0;
  for(int i = 0; i < num_particles; ++i) {
    //calculate values just once
    const double theta = particles[i].theta;
    const double sin_t = sin(theta);
    const double cos_t = cos(theta);
    
    vector<LandmarkObs> observations_on_map;
    for(LandmarkObs observation: observations) {
      LandmarkObs obs;
      //Transforming observations to map coordinate system
      obs.id = observation.id;
      obs.x = particles[i].x + cos_t * observation.x - sin_t * observation.y;
      obs.y = particles[i].y + sin_t * observation.x + cos_t * observation.y;
      observations_on_map.push_back(obs);
    }
    
    //filter unnecessary map_landmarks
    vector<LandmarkObs> landmarks_in_range;
    for (Map::single_landmark_s landmark: map_landmarks.landmark_list) {
      const double landmarkDist = sqrt(pow(landmark.x_f - particles[i].x, 2) + pow(landmark.y_f - particles[i].y, 2));
      if (landmarkDist <= sensor_range) {
        LandmarkObs obs;
        obs.id = landmark.id_i;
        obs.x = landmark.x_f;
        obs.y = landmark.y_f;
        landmarks_in_range.push_back(obs);
      }
    }
    
    //Uncomment to debug
    //if(landmarks_in_range.size() < observations_on_map.size()) {
    //  cout << "More observations than landmarks in range of sensor for particle " << i << endl;
    //}
    
    dataAssociation(landmarks_in_range, observations_on_map);
    
    //calculate particles final weight
    const double std_x = std_landmark[0];
    const double std_y = std_landmark[1];
    double temp_weight = 1.0;
    
    //Comment if you don't need associations
    //https://discussions.udacity.com/t/what-should-i-write-in-associations-sense-x-and-sens-y-of-particle-objects/295671/2
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    
    const double basePart = 1 / (2 * M_PI * std_x * std_y);
    const double sigma_2x2 = 2 * pow(std_x, 2);
    const double sigma_2y2 = 2 * pow(std_y, 2);
    
    for(LandmarkObs obs: observations_on_map) {
      const double assumed_landmark_x = map_landmarks.landmark_list[obs.id - 1].x_f;
      const double assumed_landmark_y = map_landmarks.landmark_list[obs.id - 1].y_f;
      
      const double c0 = pow(obs.x - assumed_landmark_x, 2) / sigma_2x2;
      const double c1 = pow(obs.y - assumed_landmark_y, 2) / sigma_2y2;
      const double observation_weight = basePart * exp(-(c0 + c1));
      temp_weight *=  observation_weight;
    }
    particles[i].weight = temp_weight;
    weights[i] = temp_weight;
    totalWeight += temp_weight;
    //Comment if you don't need associations
    particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
  }
  
  for (int i = 0; i < num_particles; ++i) {
    const double particleWeight = particles[i].weight / totalWeight;
    particles[i].weight = particleWeight;
    weights[i] = particleWeight;
    
    //Comment if you don't need associations
    particles[i] = SetAssociations(particles[i], particles[i].associations, particles[i].sense_x, particles[i].sense_y);
    //Uncomment to debug
    //cout << "normalized particle " << i << " (" << particles[i].x << "," << particles[i].y << ") weight: " << particleWeight << endl;
  }
  
}

void ParticleFilter::resample() {
  
  vector<Particle> new_list;
  default_random_engine generator;
  discrete_distribution<> distribution (weights.begin(), weights.end());
  for (int i = 0; i < num_particles; ++i) {
    const int idx = distribution(generator);
    //Comment if you don't need associations
    particles[idx] = SetAssociations(particles[idx], particles[idx].associations, particles[idx].sense_x, particles[idx].sense_y);
    new_list.push_back(particles[idx]);
  }
  
  particles.clear();
  particles = new_list;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  
  particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;
  
 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
