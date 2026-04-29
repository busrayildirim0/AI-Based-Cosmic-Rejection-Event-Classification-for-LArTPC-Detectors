#ifndef EventInformation_h
#define EventInformation_h 1

#include "G4VUserEventInformation.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"

class EventInformation : public G4VUserEventInformation
{
public:
    EventInformation();
    virtual ~EventInformation();
    
    virtual void Print() const;
    
    void SetNeutrinoKinematics(G4double Q2, G4double W, G4double Enu, G4String type);
    
    G4double GetQ2() const { return fQ2; }
    G4double GetW() const { return fW; }
    G4double GetEnu() const { return fEnu; }
    G4String GetInteractionType() const { return fInteractionType; }
    
    void SetPrimaryVertex(const G4ThreeVector& vertex) { fPrimaryVertex = vertex; }
    void SetPrimaryMomentum(const G4ThreeVector& momentum) { fPrimaryMomentum = momentum; }
    void SetPrimaryEnergy(G4double energy) { fPrimaryEnergy = energy; }
    void SetPrimaryPDG(G4int pdg) { fPrimaryPDG = pdg; }
    
    G4ThreeVector GetPrimaryVertex() const { return fPrimaryVertex; }
    G4ThreeVector GetPrimaryMomentum() const { return fPrimaryMomentum; }
    G4double GetPrimaryEnergy() const { return fPrimaryEnergy; }
    G4int GetPrimaryPDG() const { return fPrimaryPDG; }
    
    G4double GetPrimaryMomentumMag() const { return fPrimaryMomentum.mag(); }
    G4double GetPrimaryTheta() const; 
    G4double GetPrimaryPhi() const;   
    
private:
    G4double fQ2;                
    G4double fW;                 
    G4double fEnu;               
    G4String fInteractionType;   
    
    G4ThreeVector fPrimaryVertex;   
    G4ThreeVector fPrimaryMomentum; 
    G4double fPrimaryEnergy;        
    G4int fPrimaryPDG;              
};

#endif