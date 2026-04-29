#ifndef LArSensitiveDetector_h
#define LArSensitiveDetector_h 1

#include "G4VSensitiveDetector.hh"
#include "LArHit.hh"
#include "DetectorConstruction.hh"
#include <vector>

class G4Step;
class G4HCofThisEvent;

class LArSensitiveDetector : public G4VSensitiveDetector
{
public:
    LArSensitiveDetector(const G4String& name, const G4String& hitsCollectionName);
    virtual ~LArSensitiveDetector();

    virtual void Initialize(G4HCofThisEvent* hitCollection);
    virtual G4bool ProcessHits(G4Step* step, G4TouchableHistory* history);
    virtual void EndOfEvent(G4HCofThisEvent* hitCollection);

private:
    LArHitsCollection* fHitsCollection;
    G4int fHCID;

    struct WireResponse {
        G4int wireNumber;
        G4int planeID;      
        G4double signal;    
        G4double time;      
    };
    
    std::vector<WireResponse> CalculateWireResponse(const G4ThreeVector& position,
                                                   G4double energy,
                                                   G4double time);
    
    G4int CalculateWireNumber(const G4ThreeVector& position, G4int planeID);
    G4double CalculateDriftTime(const G4ThreeVector& position, G4int planeID);
    G4double CalculateSignalStrength(const G4ThreeVector& position, G4double energy, G4int planeID);
    
    static constexpr G4double fWireSpacing = DetectorConstruction::fWireSpacing; 
    static constexpr G4double fDriftVelocity = DetectorConstruction::fDriftVelocity; 
    
    static constexpr G4int fNWires_U = G4int(DetectorConstruction::fTPC_Y / fWireSpacing) + 100; 
    static constexpr G4int fNWires_V = G4int(DetectorConstruction::fTPC_Y / fWireSpacing) + 100;
    static constexpr G4int fNWires_Y = G4int(DetectorConstruction::fTPC_Y / fWireSpacing) + 100;
};

#endif