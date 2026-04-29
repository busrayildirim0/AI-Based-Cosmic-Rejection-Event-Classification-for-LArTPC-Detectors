#include "LArHit.hh"
#include "G4UnitsTable.hh"
#include "G4VVisManager.hh"
#include "G4Circle.hh"
#include "G4Colour.hh"
#include "G4VisAttributes.hh"
#include <iomanip>

G4ThreadLocal G4Allocator<LArHit>* LArHitAllocator = nullptr;

LArHit::LArHit()
    : G4VHit(),
      fWirePlaneID(-1),
      fWireNumber(-1),
      fEnergyDeposit(0.),
      fTime(0.),
      fPosition(G4ThreeVector()),
      fParticlePDG(0),
      fTrackID(-1),
      fParentTrackID(-1),
      fProcessName(""),
      fScintillationPhotons(0)
{}

LArHit::~LArHit() {}

LArHit::LArHit(const LArHit& right)
    : G4VHit()
{
    fWirePlaneID = right.fWirePlaneID;
    fWireNumber = right.fWireNumber;
    fEnergyDeposit = right.fEnergyDeposit;
    fTime = right.fTime;
    fPosition = right.fPosition;
    fParticlePDG = right.fParticlePDG;
    fTrackID = right.fTrackID;
    fParentTrackID = right.fParentTrackID;
    fProcessName = right.fProcessName;
    fScintillationPhotons = right.fScintillationPhotons;
}

const LArHit& LArHit::operator=(const LArHit& right)
{
    fWirePlaneID = right.fWirePlaneID;
    fWireNumber = right.fWireNumber;
    fEnergyDeposit = right.fEnergyDeposit;
    fTime = right.fTime;
    fPosition = right.fPosition;
    fParticlePDG = right.fParticlePDG;
    fTrackID = right.fTrackID;
    fParentTrackID = right.fParentTrackID;
    fProcessName = right.fProcessName;
    fScintillationPhotons = right.fScintillationPhotons;

    return *this;
}

G4bool LArHit::operator==(const LArHit& right) const
{
    return ( this == &right ) ? true : false;
}

void LArHit::Draw()
{
    G4VVisManager* pVVisManager = G4VVisManager::GetConcreteInstance();
    if(pVVisManager)
    {
        G4Circle circle(fPosition);
        circle.SetScreenSize(4.);
        circle.SetFillStyle(G4Circle::filled);

        G4Colour colour;
        switch(fWirePlaneID) {
            case 0: colour = G4Colour::Red(); break;
            case 1: colour = G4Colour::Green(); break;
            case 2: colour = G4Colour::Blue(); break;
            default: colour = G4Colour::White(); break;
        }

        G4VisAttributes attribs(colour);
        circle.SetVisAttributes(attribs);
        pVVisManager->Draw(circle);
    }
}

void LArHit::Print()
{
    G4cout << std::setprecision(3) << std::fixed
           << "Wire Plane: " << fWirePlaneID;
    
    switch(fWirePlaneID) {
        case 0: G4cout << "(U)"; break;
        case 1: G4cout << "(V)"; break;
        case 2: G4cout << "(Y)"; break;
        default: G4cout << "(?)"; break;
    }
    
    G4cout << ", Wire: " << std::setw(4) << fWireNumber
           << ", Energy: " << std::setw(8) << G4BestUnit(fEnergyDeposit,"Energy")
           << ", Time: " << std::setw(8) << G4BestUnit(fTime, "Time")
           << ", Pos: (" << std::setw(6) << fPosition.x()/CLHEP::cm 
           << "," << std::setw(6) << fPosition.y()/CLHEP::cm
           << "," << std::setw(6) << fPosition.z()/CLHEP::cm << ") cm"
           << ", PDG: " << std::setw(5) << fParticlePDG
           << ", TrackID: " << fTrackID;
    
    if (fParentTrackID != 0) {
        G4cout << ", Parent: " << fParentTrackID;
    }
    
    if (!fProcessName.empty()) {
        G4cout << ", Process: " << fProcessName;
    }
    
    G4cout << G4endl;
}
