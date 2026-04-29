#ifndef LArHit_h
#define LArHit_h 1

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include "G4Threading.hh"

class LArHit : public G4VHit
{
public:
    LArHit();
    LArHit(const LArHit&);
    virtual ~LArHit();

    const LArHit& operator=(const LArHit&);
    G4bool operator==(const LArHit&) const;

    inline void* operator new(size_t);
    inline void operator delete(void*);

    virtual void Draw();
    virtual void Print();

    void SetWirePlaneID(G4int id) { fWirePlaneID = id; }
    void SetWireNumber(G4int wireNum) { fWireNumber = wireNum; }
    void SetEnergyDeposit(G4double edep) { fEnergyDeposit = edep; }
    void SetTime(G4double time) { fTime = time; }
    void SetPosition(G4ThreeVector pos) { fPosition = pos; }
    void SetParticleType(G4int pdg) { fParticlePDG = pdg; }
    void SetTrackID(G4int id) { fTrackID = id; }
    void SetParentTrackID(G4int id) { fParentTrackID = id; }
    void SetProcessName(G4String process) { fProcessName = process; }
    void SetScintillationPhotons(G4int photons) { fScintillationPhotons = photons; }

    G4int GetWirePlaneID() const { return fWirePlaneID; }
    G4int GetWireNumber() const { return fWireNumber; }
    G4double GetEnergyDeposit() const { return fEnergyDeposit; }
    G4double GetTime() const { return fTime; }
    G4ThreeVector GetPosition() const { return fPosition; }
    G4int GetParticleType() const { return fParticlePDG; }
    G4int GetTrackID() const { return fTrackID; }
    G4int GetParentTrackID() const { return fParentTrackID; }
    G4String GetProcessName() const { return fProcessName; }
    G4int GetScintillationPhotons() const { return fScintillationPhotons; }

    G4bool IsFromPrimary() const { return fParentTrackID == 0; }
    G4bool IsInCollectionPlane() const { return fWirePlaneID == 2; }
    G4bool IsInInductionPlane() const { return fWirePlaneID < 2; }

private:
    G4int fWirePlaneID;           
    G4int fWireNumber;            
    G4double fEnergyDeposit;     
    G4double fTime;               
    G4ThreeVector fPosition;      
    G4int fParticlePDG;          
    G4int fTrackID;              
    G4int fParentTrackID;        
    G4String fProcessName;        
    G4int fScintillationPhotons; 
};

typedef G4THitsCollection<LArHit> LArHitsCollection;

extern G4ThreadLocal G4Allocator<LArHit>* LArHitAllocator;

inline void* LArHit::operator new(size_t)
{
    if(!LArHitAllocator)
        LArHitAllocator = new G4Allocator<LArHit>;
    return (void *) LArHitAllocator->MallocSingle();
}

inline void LArHit::operator delete(void *hit)
{
    LArHitAllocator->FreeSingle((LArHit*) hit);
}

#endif