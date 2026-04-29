#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"
#include <vector>

class RunAction;
class G4Event;

class EventAction : public G4UserEventAction
{
public:
    EventAction(RunAction* runAction);
    virtual ~EventAction();

    virtual void BeginOfEventAction(const G4Event* event);
    virtual void EndOfEventAction(const G4Event* event);

    void AddEnergyDeposit(G4double edep) { fEnergyDeposit += edep; }
    void AddTrackLength(G4double length) { fTrackLength += length; }
    
    void SetNeutrinoKinematics(G4double Q2, G4double W, G4double Enu, G4String type);
    G4double GetNeutrinoQ2() const { return fNeutrinoQ2; }
    G4double GetNeutrinoW() const { return fNeutrinoW; }
    G4double GetNeutrinoEnu() const { return fNeutrinoEnu; }
    G4String GetInteractionType() const { return fInteractionType; }
    
private:
    void CollectHitData(const G4Event* event);
    void CalculateGeometricFeatures();  
    void SaveEventData();
    void ResetEventData();
    
    void CalculatePCAFeatures();           
    void CalculateDeDxProfile();           
    void CalculateHitDensity();            
    void CalculateWirePlaneHits();         
    void CalculateOpeningAngle();          
    
    RunAction* fRunAction;
    
    G4double fEnergyDeposit;
    G4double fTrackLength;
    G4int fEventID;
    G4int fEventType; 
    
    G4double fNeutrinoQ2;        
    G4double fNeutrinoW;         
    G4double fNeutrinoEnu;      
    G4String fInteractionType;   
    
    std::vector<G4double> fHitX, fHitY, fHitZ;
    std::vector<G4double> fHitEnergy;
    std::vector<G4double> fHitTime;
    std::vector<G4int> fWirePlane;
    std::vector<G4int> fWireNumber;
    std::vector<G4int> fParticlePDG;
    
    G4double fVertexX, fVertexY, fVertexZ;           
    G4double fPrimaryPx, fPrimaryPy, fPrimaryPz;     
    G4double fPrimaryE;                              
    G4double fPrimaryP;                              
    G4double fPrimaryTheta;                          
    G4double fPrimaryPhi;                            
    G4int fPrimaryPDG;                               
    
    G4int fNHits;                                    
    G4double fVisibleEnergy;                         
    G4double fAvgHitEnergy;                          
    
    G4double fHitLengthX;                            
    G4double fHitLengthY;                            
    G4double fHitLengthZ;                            
    G4double fHitAspectRatio;                        
    
    G4double fPCAEigenvalue1;                        
    G4double fPCAEigenvalue2;                        
    G4double fPCAEigenvalue3;                        
    G4double fPCARatio12;                            
    G4double fPCARatio13;                            
    
    G4double fOpeningAngle;                          
    G4double fTrackAngleWrtBeam;                     
    
    G4double fDeDxMean;                              
    G4double fDeDxStd;                               
    G4double fEnergyFrontFraction;                   
    G4double fEnergyBackFraction;                    
    
    G4double fHitDensity;                            
    G4double fHitTimeSpread;                         
    G4int fNIsolatedHits;                            
    
    G4int fNHitsU;                                   
    G4int fNHitsV;                                   
    G4int fNHitsY;                                   
    G4double fHitPlaneRatioYUV;                      
};

#endif