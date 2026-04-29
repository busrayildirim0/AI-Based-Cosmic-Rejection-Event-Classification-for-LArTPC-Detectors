#include "EventInformation.hh"
#include "G4ios.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>

EventInformation::EventInformation()
    : G4VUserEventInformation(),
      fQ2(-1.),
      fW(-1.),
      fEnu(-1.),
      fInteractionType("Unknown"),
      fPrimaryVertex(G4ThreeVector(0., 0., 0.)),
      fPrimaryMomentum(G4ThreeVector(0., 0., 0.)),
      fPrimaryEnergy(0.),
      fPrimaryPDG(0)
{
}

EventInformation::~EventInformation()
{
}

void EventInformation::SetNeutrinoKinematics(G4double Q2, G4double W, G4double Enu, G4String type)
{
    fQ2 = Q2;
    fW = W;
    fEnu = Enu;
    fInteractionType = type;
}

G4double EventInformation::GetPrimaryTheta() const
{
    G4double p_mag = fPrimaryMomentum.mag();
    if (p_mag > 0) {
        G4double cos_theta = fPrimaryMomentum.z() / p_mag;
        if (cos_theta > 1.0) cos_theta = 1.0;
        if (cos_theta < -1.0) cos_theta = -1.0;
        return std::acos(cos_theta) * 180.0 / M_PI;
    }
    return 0.;
}

G4double EventInformation::GetPrimaryPhi() const
{
    G4double phi = std::atan2(fPrimaryMomentum.y(), fPrimaryMomentum.x());
    return phi * 180.0 / M_PI;
}

void EventInformation::Print() const
{
    G4cout << "========================================" << G4endl;
    G4cout << "EventInformation:" << G4endl;
    G4cout << "  Interaction Type: " << fInteractionType << G4endl;
    
    if (fInteractionType != "Cosmic") {
        G4cout << "  Neutrino Kinematics:" << G4endl;
        G4cout << "    Enu = " << fEnu << " GeV" << G4endl;
        G4cout << "    Q2  = " << fQ2 << " GeV²" << G4endl;
        G4cout << "    W   = " << fW << " GeV" << G4endl;
    }
    
    G4cout << "  Primary vertex (cm): ("
           << fPrimaryVertex.x()/cm << ", "
           << fPrimaryVertex.y()/cm << ", "
           << fPrimaryVertex.z()/cm << ")" << G4endl;

    G4cout << "  Primary momentum (GeV/c): ("
           << fPrimaryMomentum.x() << ", "
           << fPrimaryMomentum.y() << ", "
           << fPrimaryMomentum.z() << ")" << G4endl;

    G4cout << "  Primary Energy: " << fPrimaryEnergy << " GeV" << G4endl;
    G4cout << "  Primary |p|: " << GetPrimaryMomentumMag() << " GeV/c" << G4endl;
    G4cout << "  Primary theta: " << GetPrimaryTheta() << "°" << G4endl;
    G4cout << "  Primary phi: " << GetPrimaryPhi() << "°" << G4endl;
    G4cout << "  Primary PDG: " << fPrimaryPDG << G4endl;
    G4cout << "========================================" << G4endl;
}