// PrimaryGeneratorAction.hh

#ifndef PrimaryGeneratorAction_h
#define PrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "globals.hh"

class G4ParticleGun;
class G4Event;
class EventInformation;
class PrimaryGeneratorMessenger;  

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
    enum GeneratorMode {
        kCosmicRayMode,
        kNeutrinoMode,
        kTestMode
    };

    PrimaryGeneratorAction();    
    virtual ~PrimaryGeneratorAction();

    virtual void GeneratePrimaries(G4Event*);

    void SetGeneratorMode(GeneratorMode mode) { fGeneratorMode = mode; }
    GeneratorMode GetGeneratorMode() const { return fGeneratorMode; }

    void SetCosmicEnergyRange(G4double minE, G4double maxE);
    void SetCosmicAngleRange(G4double minTheta, G4double maxTheta);
    void SetNeutrinoEnergyRange(G4double minE, G4double maxE);

private:
    void GenerateCosmicRay(G4Event* anEvent, EventInformation* eventInfo);
    void GenerateNeutrinoEvent(G4Event* anEvent, EventInformation* eventInfo);
    void GenerateTestParticle(G4Event* anEvent);

    void GenerateCCQEEvent(G4Event* anEvent, EventInformation* eventInfo, G4double Enu);
    void GenerateCC1PiEvent(G4Event* anEvent, EventInformation* eventInfo, G4double Enu);
    void GenerateDISEvent(G4Event* anEvent, EventInformation* eventInfo, G4double Enu);

    G4double SampleCosmicEnergy();
    G4double SampleCosmicAngle();
    G4ThreeVector SampleCosmicPosition();
    
    G4ThreeVector FindLArEntryPoint(G4ThreeVector position, G4ThreeVector direction);

    void InitializeBNBFlux();
    G4double SampleBNBEnergy();
    G4double SampleCCQE_Q2(G4double Enu);
    G4double SampleResonance_W();
    G4double SampleDIS_y(G4double Enu);

    G4ParticleGun* fParticleGun;
    GeneratorMode fGeneratorMode;

    G4double fCosmicMinEnergy;
    G4double fCosmicMaxEnergy;
    G4double fCosmicMinTheta;
    G4double fCosmicMaxTheta;

    G4double fNeutrinoMinEnergy;
    G4double fNeutrinoMaxEnergy;

    G4int fCosmicEventCount;
    G4int fNeutrinoEventCount;

    static const G4int fNBNBBins = 60;
    G4double fBNBEnergyBins[61];
    G4double fBNBFluxWeights[60];
    G4double fBNBCumulativeWeights[60];
    
    PrimaryGeneratorMessenger* fMessenger; 
};

#endif
