#ifndef RunAction_h
#define RunAction_h 1

#include "G4UserRunAction.hh"
#include "G4Accumulable.hh"
#include "globals.hh"
#include <fstream>

class G4Run;

class RunAction : public G4UserRunAction
{
public:
    RunAction();
    virtual ~RunAction();
    
    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);
    
    void AddEnergyDeposit(G4double edep);
    void AddTrackLength(G4double length);
    void IncrementEventType(G4int eventType); 
    
    void InitializeDataOutput();
    void FinalizeDataOutput();
    
    
    void WriteEventData(
        
        G4int eventID,
        G4int eventType,
        const G4String& interactionType,
        
        G4double neutrinoEnu,
        G4double neutrinoQ2,
        G4double neutrinoW,
        
        G4double vertexX,
        G4double vertexY,
        G4double vertexZ,
        
        G4double primaryPx,
        G4double primaryPy,
        G4double primaryPz,
        G4double primaryP,
        G4double primaryE,
        G4double primaryTheta,
        G4double primaryPhi,
        G4int primaryPDG,
        
        G4int nHits,
        G4double visibleEnergy,
        G4double avgHitEnergy,
        
        G4double hitLengthX,
        G4double hitLengthY,
        G4double hitLengthZ,
        G4double hitAspectRatio,
        
        G4double pca1,
        G4double pca2,
        G4double pca3,
        G4double pcaRatio12,
        G4double pcaRatio13,
        
        
        G4double openingAngle,
        G4double trackAngleWrtBeam,
        
        G4double dEdxMean,
        G4double dEdxStd,
        G4double energyFrontFraction,
        G4double energyBackFraction,
        
        G4double hitDensity,
        G4double hitTimeSpread,
        G4int nIsolatedHits,
        
        G4int nHitsU,
        G4int nHitsV,
        G4int nHitsY,
        G4double hitPlaneRatioYUV
    );
    
private:
    G4Accumulable<G4double> fEnergyDeposit;
    G4Accumulable<G4double> fTrackLength;
    G4Accumulable<G4int> fCosmicEvents;
    G4Accumulable<G4int> fNeutrinoEvents;
    G4Accumulable<G4int> fTestEvents;
    
    G4String fOutputFileName;
    G4bool fDataOutputInitialized;
    std::ofstream fDataFile;
};

#endif
