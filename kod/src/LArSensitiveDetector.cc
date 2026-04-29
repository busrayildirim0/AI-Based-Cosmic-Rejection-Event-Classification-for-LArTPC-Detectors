#include "LArSensitiveDetector.hh"
#include "G4HCofThisEvent.hh"
#include "G4Step.hh"
#include "G4ThreeVector.hh"
#include "G4SDManager.hh"
#include "G4ios.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4ParticleDefinition.hh"
#include <cmath>

LArSensitiveDetector::LArSensitiveDetector(const G4String& name, const G4String& hitsCollectionName)
    : G4VSensitiveDetector(name),
      fHitsCollection(nullptr),
      fHCID(-1)
{
    collectionName.insert(hitsCollectionName);
    G4cout << "LArSensitiveDetector constructed: " << name << G4endl;
}

LArSensitiveDetector::~LArSensitiveDetector()
{}

void LArSensitiveDetector::Initialize(G4HCofThisEvent* hce)
{
    fHitsCollection = new LArHitsCollection(SensitiveDetectorName, collectionName[0]);
    
    if (fHCID < 0) {
        fHCID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
    }

    hce->AddHitsCollection(fHCID, fHitsCollection);
}

G4bool LArSensitiveDetector::ProcessHits(G4Step* step, G4TouchableHistory*)
{
    G4double edep = step->GetTotalEnergyDeposit();
    if (edep == 0.) return false;

    G4StepPoint* preStepPoint = step->GetPreStepPoint();
    G4ThreeVector position = preStepPoint->GetPosition();
    G4double time = preStepPoint->GetGlobalTime();

    G4Track* track = step->GetTrack();
    G4int pdgCode = track->GetDefinition()->GetPDGEncoding();
    G4int trackID = track->GetTrackID();
    G4int parentID = track->GetParentID();

    G4String processName = "";
    const G4VProcess* process = step->GetPostStepPoint()->GetProcessDefinedStep();
    if (process) processName = process->GetProcessName();

    std::vector<WireResponse> wireResponses = CalculateWireResponse(position, edep, time);

    for (const auto& response : wireResponses) {
        if (response.signal > 0.001 * MeV) {
            LArHit* newHit = new LArHit();

            newHit->SetWirePlaneID(response.planeID);
            newHit->SetWireNumber(response.wireNumber);
            newHit->SetEnergyDeposit(response.signal);
            newHit->SetTime(response.time);
            newHit->SetPosition(position);
            newHit->SetParticleType(pdgCode);
            newHit->SetTrackID(trackID);
            newHit->SetParentTrackID(parentID);
            newHit->SetProcessName(processName);

            G4int photons = static_cast<G4int>(edep / MeV * 40000);
            newHit->SetScintillationPhotons(photons);

            fHitsCollection->insert(newHit);
        }
    }

    return true;
}

std::vector<LArSensitiveDetector::WireResponse>
LArSensitiveDetector::CalculateWireResponse(const G4ThreeVector& position, G4double energy, G4double time)
{
    std::vector<WireResponse> responses;

    for (G4int planeID = 0; planeID < 3; planeID++) {
        G4int wireNumber = CalculateWireNumber(position, planeID);
        G4double driftTime = CalculateDriftTime(position, planeID);
        G4double signalStrength = CalculateSignalStrength(position, energy, planeID);

        G4int maxWires = (planeID == 0) ? fNWires_U :
                         (planeID == 1) ? fNWires_V :
                                          fNWires_Y;

        if (wireNumber >= 0 && wireNumber < maxWires) {
            WireResponse response;
            response.wireNumber = wireNumber;
            response.planeID = planeID;
            response.signal = signalStrength;
            response.time = time + driftTime;
            responses.push_back(response);
        }
    }

    return responses;
}

G4int LArSensitiveDetector::CalculateWireNumber(const G4ThreeVector& position, G4int planeID)
{
    G4double x = position.x() / cm;
    G4double y = position.y() / cm;
    G4double z = position.z() / cm;

    G4int wireNumber = -1;

    switch(planeID) {
        case 0:
        {
            G4double wireCoord = x * cos(30.*deg) + y * sin(30.*deg);
            wireNumber = static_cast<G4int>((wireCoord + DetectorConstruction::fTPC_Y/2.) / fWireSpacing);
            break;
        }
        case 1:
        {
            G4double wireCoord = -x * cos(30.*deg) + y * sin(30.*deg);
            wireNumber = static_cast<G4int>((wireCoord + DetectorConstruction::fTPC_Y/2.) / fWireSpacing);
            break;
        }
        case 2:
        {
            wireNumber = static_cast<G4int>((y + DetectorConstruction::fTPC_Y/2.) / fWireSpacing);
            break;
        }
    }

    return wireNumber;
}

G4double LArSensitiveDetector::CalculateDriftTime(const G4ThreeVector& position, G4int planeID)
{
    G4double z = position.z() / cm;
    G4double driftDistance = 0.;

    switch(planeID) {
        case 0:
            driftDistance = std::abs(z - DetectorConstruction::fWirePlaneU_Z);
            break;
        case 1:
            driftDistance = std::abs(z - DetectorConstruction::fWirePlaneV_Z);
            break;
        case 2:
            driftDistance = std::abs(z - DetectorConstruction::fWirePlaneY_Z);
            break;
    }

    G4double driftTime = driftDistance / fDriftVelocity;
    return driftTime * microsecond;
}

G4double LArSensitiveDetector::CalculateSignalStrength(const G4ThreeVector& position, G4double energy, G4int planeID)
{
    G4double signalFactor = 1.0;

    if (planeID != 2) {
        signalFactor = 0.3;

        G4double z = position.z() / cm;
        G4double wirePlaneZ = (planeID == 0) ? DetectorConstruction::fWirePlaneU_Z :
                             (planeID == 1) ? DetectorConstruction::fWirePlaneV_Z :
                                              0;

        G4double distance = std::abs(z - wirePlaneZ);
        signalFactor *= exp(-distance / 50.);
    }

    return energy * signalFactor;
}

void LArSensitiveDetector::EndOfEvent(G4HCofThisEvent*)
{
    if (verboseLevel > 1) {
        G4int nofHits = fHitsCollection->entries();
        G4cout << G4endl
               << "-------->Hits Collection: " << nofHits
               << " hits in the LAr TPC:" << G4endl;

        G4int maxPrint = std::min(nofHits, 10);
        for (G4int i = 0; i < maxPrint; i++) {
            (*fHitsCollection)[i]->Print();
        }
        if (nofHits > maxPrint) {
            G4cout << "... and " << (nofHits - maxPrint) << " more hits" << G4endl;
        }
    }
}
