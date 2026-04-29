#include "PrimaryGeneratorAction.hh"
#include "DetectorConstruction.hh"
#include "EventInformation.hh"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4RunManager.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"
#include "G4RandomDirection.hh"

#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4Proton.hh"
#include "G4Neutron.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"

#include "G4PionZero.hh" 
#include "G4RandomDirection.hh"
#include <cmath>

PrimaryGeneratorAction::PrimaryGeneratorAction()
    : G4VUserPrimaryGeneratorAction(),
      fParticleGun(nullptr),
      //
      fGeneratorMode(kNeutrinoMode),
      //
      //fGeneratorMode(kCosmicRayMode),
      fCosmicMinEnergy(0.5*GeV),
      fCosmicMaxEnergy(1000.*GeV),
      fCosmicMinTheta(0.*deg),
      fCosmicMaxTheta(60.*deg),
      fNeutrinoMinEnergy(0.2*GeV),
      fNeutrinoMaxEnergy(5.*GeV),
      fCosmicEventCount(0),
      fNeutrinoEventCount(0)
{
    G4int n_particle = 1;
    fParticleGun = new G4ParticleGun(n_particle);

    G4ParticleDefinition* particle = G4MuonMinus::MuonMinus();
    fParticleGun->SetParticleDefinition(particle);
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.,0.,-1.));
    fParticleGun->SetParticleEnergy(4.*GeV);
    
    InitializeBNBFlux();
    
    G4cout << "PrimaryGeneratorAction constructed" << G4endl;
    if (fGeneratorMode == kCosmicRayMode) {
        G4cout << "Default mode: Cosmic Ray (realistic physics)" << G4endl;
    } else {
        G4cout << "Default mode: Neutrino with BNB flux (MicroBooNE official data)" << G4endl;
    }
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fParticleGun;
    
    G4cout << "Event generation summary:" << G4endl;
    G4cout << "  Cosmic ray events: " << fCosmicEventCount << G4endl;
    G4cout << "  Neutrino events: " << fNeutrinoEventCount << G4endl;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
    EventInformation* eventInfo = new EventInformation();
    anEvent->SetUserInformation(eventInfo);
    
    switch(fGeneratorMode) {
        case kCosmicRayMode:
            GenerateCosmicRay(anEvent, eventInfo);
            fCosmicEventCount++;
            break;
            
        case kNeutrinoMode:
            GenerateNeutrinoEvent(anEvent, eventInfo);
            fNeutrinoEventCount++;
            break;
            
        case kTestMode:
            GenerateTestParticle(anEvent);
            break;
    }
}

void PrimaryGeneratorAction::GenerateCosmicRay(G4Event* anEvent, EventInformation* eventInfo)
{
    G4double energy = SampleCosmicEnergy();
    G4double theta = SampleCosmicAngle();
    G4double phi = G4UniformRand() * 2.0 * pi;
    
    G4ThreeVector direction(sin(theta)*cos(phi), sin(theta)*sin(phi), -cos(theta));
    G4ThreeVector initialPosition = SampleCosmicPosition();
    
    G4ThreeVector entryPoint = FindLArEntryPoint(initialPosition, direction);
    
    G4ParticleDefinition* particle = nullptr;
    G4double randParticle = G4UniformRand();
    
    if (randParticle < 0.79) {
        if (G4UniformRand() < 0.5) {
            particle = G4MuonMinus::MuonMinus();
        } else {
            particle = G4MuonPlus::MuonPlus();
        }
    } else if (randParticle < 0.92) {
        particle = G4Proton::Proton();
    } else if (randParticle < 0.99) {
        particle = G4Neutron::Neutron();
    } else {
        if (G4UniformRand() < 0.5) {
            particle = G4Electron::Electron();
        } else {
            particle = G4Positron::Positron();
        }
    }
    
    fParticleGun->SetParticleDefinition(particle);
    fParticleGun->SetParticleEnergy(energy);
    fParticleGun->SetParticleMomentumDirection(direction);
    fParticleGun->SetParticlePosition(entryPoint);
    fParticleGun->GeneratePrimaryVertex(anEvent);
    
    G4double p_mag = sqrt(energy*energy + 2*energy*particle->GetPDGMass());
    G4ThreeVector p_vec = direction * p_mag;
    
    eventInfo->SetPrimaryVertex(entryPoint);
    eventInfo->SetPrimaryMomentum(p_vec/GeV);
    eventInfo->SetPrimaryEnergy((energy + particle->GetPDGMass())/GeV);
    eventInfo->SetPrimaryPDG(particle->GetPDGEncoding());
    eventInfo->SetNeutrinoKinematics(-1, -1, -1, "Cosmic");
}

void PrimaryGeneratorAction::GenerateNeutrinoEvent(G4Event* anEvent, EventInformation* eventInfo)
{
    G4double Enu = SampleBNBEnergy();
    
    G4double rand = G4UniformRand();
    
    if (rand < 0.45) {
        GenerateCCQEEvent(anEvent, eventInfo, Enu);
    } else if (rand < 0.75) {
        GenerateCC1PiEvent(anEvent, eventInfo, Enu);
    } else {
        GenerateDISEvent(anEvent, eventInfo, Enu);
    }
}

void PrimaryGeneratorAction::GenerateCCQEEvent(G4Event* anEvent, EventInformation* eventInfo, G4double Enu)
{
    G4double x = (G4UniformRand() - 0.5) * DetectorConstruction::fTPC_X * cm;
    G4double y = (G4UniformRand() - 0.5) * DetectorConstruction::fTPC_Y * cm;
    G4double z = (G4UniformRand() - 0.5) * DetectorConstruction::fTPC_Z * cm;
    G4ThreeVector position(x, y, z);
    
    G4double m_mu = 0.1057*GeV;
    G4double M_n = 0.939*GeV;
    G4double M_p = 0.938*GeV;
    
    G4double Q2 = SampleCCQE_Q2(Enu);
    
    G4double p_fermi = 0.220*GeV;
    G4ThreeVector pF_vec = G4RandomDirection() * p_fermi;
    G4double EF = sqrt(M_n*M_n + p_fermi*p_fermi);
    
    G4double Eb = 27*MeV;
    
    G4double Eavail = Enu + EF - Eb;
    G4double Emu = Eavail - Q2/(2.0*M_n);
    
    if (Emu < m_mu + 50*MeV) Emu = m_mu + 50*MeV;
    if (Emu > Enu - 50*MeV) Emu = Enu - 50*MeV;
    
    G4double cos_theta = 1.0 - (Q2 + m_mu*m_mu)/(2.0*Enu*Emu);
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;
    
    G4double theta_mu = acos(cos_theta);
    G4double phi_mu = G4UniformRand() * 2.0 * pi;
    
    G4ThreeVector mu_dir(sin(theta_mu)*cos(phi_mu), 
                         sin(theta_mu)*sin(phi_mu), 
                         cos(theta_mu));
    
    G4double KEmu = Emu - m_mu;
    if (KEmu <= 0) KEmu = 50*MeV;
    
    fParticleGun->SetParticleDefinition(G4MuonMinus::MuonMinus());
    fParticleGun->SetParticleEnergy(KEmu);
    fParticleGun->SetParticleMomentumDirection(mu_dir);
    fParticleGun->SetParticlePosition(position);
    fParticleGun->GeneratePrimaryVertex(anEvent);
    
    G4double p_mu_sq = Emu*Emu - m_mu*m_mu;
    G4double p_mu;
    if (p_mu_sq > 0) {
        p_mu = sqrt(p_mu_sq);
    } else {
        p_mu = 0.05*GeV;  
    }
    G4ThreeVector p_mu_vec = mu_dir * p_mu;
    
    G4ThreeVector p_nu_vec = G4ThreeVector(0, 0, Enu);
    G4ThreeVector p_p_vec = p_nu_vec + pF_vec - p_mu_vec;
    
    G4double p_p = p_p_vec.mag();
    G4double Ep = sqrt(M_p*M_p + p_p*p_p);
    G4double KEp = Ep - M_p;
    
    if (KEp > 10*MeV && p_p > 0) {
        fParticleGun->SetParticleDefinition(G4Proton::Proton());
        fParticleGun->SetParticleEnergy(KEp);
        fParticleGun->SetParticleMomentumDirection(p_p_vec.unit());
        fParticleGun->SetParticlePosition(position);
        fParticleGun->GeneratePrimaryVertex(anEvent);
    }
    
    eventInfo->SetPrimaryVertex(position);
    eventInfo->SetPrimaryMomentum(p_mu_vec/GeV);  
    eventInfo->SetPrimaryEnergy(Emu / GeV);
    eventInfo->SetPrimaryPDG(13);  
    



    G4double W2_val = M_n*M_n + 2*M_n*(Enu-Emu) - Q2;
    G4double W;
    if (W2_val > 0.01*GeV*GeV) {
        W = sqrt(W2_val);
    } else {
        W = 0.939*GeV;
    }
    
    eventInfo->SetNeutrinoKinematics(Q2/(GeV*GeV), W/GeV, Enu/GeV, "CCQE");
}

void PrimaryGeneratorAction::GenerateCC1PiEvent(G4Event* anEvent, EventInformation* eventInfo, G4double Enu)
{
    if (Enu < 0.35*GeV) {
        GenerateCCQEEvent(anEvent, eventInfo, Enu);
        return;
    }
    
    G4double x = (G4UniformRand() - 0.5) * DetectorConstruction::fTPC_X * cm;
    G4double y = (G4UniformRand() - 0.5) * DetectorConstruction::fTPC_Y * cm;
    G4double z = (G4UniformRand() - 0.5) * DetectorConstruction::fTPC_Z * cm;
    G4ThreeVector position(x, y, z);
    
    G4double m_mu = 0.1057*GeV;
    G4double M_N = 0.939*GeV;
    
    G4double W = SampleResonance_W();
    
    G4double Q2_scale = 0.5*GeV*GeV;
    G4double Q2 = -Q2_scale * log(G4UniformRand());
    G4double Q2_max = 2.0*M_N*Enu;
    if (Q2 > Q2_max) Q2 = Q2_max * G4UniformRand();
    if (Q2 < 0.02*GeV*GeV) Q2 = 0.02*GeV*GeV;  
    


    G4double nu = (W*W - M_N*M_N + Q2) / (2.0*M_N);
    G4double Emu = Enu - nu;
    
    if (Emu < m_mu + 80*MeV) Emu = m_mu + 80*MeV;
    if (Emu > Enu - 200*MeV) Emu = Enu - 200*MeV;
    
    G4double cos_theta = 1.0 - (Q2 + m_mu*m_mu)/(2.0*Enu*Emu);
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;
    
    G4double theta_mu = acos(cos_theta);
    G4double phi_mu = G4UniformRand() * 2.0 * pi;
    
    G4ThreeVector mu_dir(sin(theta_mu)*cos(phi_mu), 
                         sin(theta_mu)*sin(phi_mu), 
                         cos(theta_mu));
    
    G4double KEmu = Emu - m_mu;
    if (KEmu <= 0) KEmu = 80*MeV;
    
    fParticleGun->SetParticleDefinition(G4MuonMinus::MuonMinus());
    fParticleGun->SetParticleEnergy(KEmu);
    fParticleGun->SetParticleMomentumDirection(mu_dir);
    fParticleGun->SetParticlePosition(position);
    fParticleGun->GeneratePrimaryVertex(anEvent);
    
    G4double p_mu_sq = Emu*Emu - m_mu*m_mu;
    G4double p_mu;
    if (p_mu_sq > 0) {
        p_mu = sqrt(p_mu_sq);
    } else {
        p_mu = 0.05*GeV;  
    }
    G4ThreeVector p_mu_vec = mu_dir * p_mu;
    
    G4double E_available = Enu - Emu;
    G4double E_remaining = E_available;
    
    G4double min_p = 80*MeV;
    G4double min_pi = 80*MeV;
    
    if (E_remaining < min_p + min_pi) {
        G4double scale = E_remaining / (min_p + min_pi);
        min_p *= scale * 0.6;
        min_pi *= scale * 0.4;
    }
    
    G4double KEp = min_p + (E_remaining - min_p - min_pi) * (0.4 + 0.3*G4UniformRand());
    if (KEp > E_remaining - min_pi - 10*MeV) KEp = E_remaining - min_pi - 10*MeV;
    E_remaining -= KEp;
    
    if (KEp > 30*MeV) {
        fParticleGun->SetParticleDefinition(G4Proton::Proton());
        fParticleGun->SetParticleEnergy(KEp);
        fParticleGun->SetParticleMomentumDirection(G4RandomDirection());
        fParticleGun->SetParticlePosition(position);
        fParticleGun->GeneratePrimaryVertex(anEvent);
    }
    
    G4double KEpi = E_remaining * (0.6 + 0.4*G4UniformRand());
    if (KEpi > E_remaining - 10*MeV) KEpi = E_remaining - 10*MeV;
    
    G4ParticleDefinition* pion;
    G4double randPion = G4UniformRand();
    if (randPion < 0.60) {
        pion = G4PionPlus::PionPlus();
    } else if (randPion < 0.90) {
        pion = G4PionZero::PionZero();
    } else {
        pion = G4PionMinus::PionMinus();
    }
    
    if (KEpi > 40*MeV) {
        fParticleGun->SetParticleDefinition(pion);
        fParticleGun->SetParticleEnergy(KEpi);
        fParticleGun->SetParticleMomentumDirection(G4RandomDirection());
        fParticleGun->SetParticlePosition(position);
        fParticleGun->GeneratePrimaryVertex(anEvent);
    }
    
    eventInfo->SetPrimaryVertex(position);
    eventInfo->SetPrimaryMomentum(p_mu_vec/GeV); 
    eventInfo->SetPrimaryEnergy(Emu / GeV);
    eventInfo->SetPrimaryPDG(13);  
    
    eventInfo->SetNeutrinoKinematics(Q2/(GeV*GeV), W/GeV, Enu/GeV, "CC1Pi");
}

void PrimaryGeneratorAction::GenerateDISEvent(G4Event* anEvent, EventInformation* eventInfo, G4double Enu)
{
    if (Enu < 0.5*GeV) {
        GenerateCCQEEvent(anEvent, eventInfo, Enu);
        return;
    }
    
    G4double pos_x = (G4UniformRand() - 0.5) * DetectorConstruction::fTPC_X * cm;
    G4double pos_y = (G4UniformRand() - 0.5) * DetectorConstruction::fTPC_Y * cm;
    G4double pos_z = (G4UniformRand() - 0.5) * DetectorConstruction::fTPC_Z * cm;
    G4ThreeVector position(pos_x, pos_y, pos_z);
    
    G4double m_mu = 0.1057*GeV;
    G4double M_N = 0.939*GeV;
    
    G4double y = SampleDIS_y(Enu);
    G4double nu = y * Enu;
    G4double Emu = Enu - nu;
    
    if (Emu < m_mu + 100*MeV) {
        Emu = m_mu + 100*MeV;
        nu = Enu - Emu;
        y = nu / Enu;
    }
    
    G4double Q2;
    if (G4UniformRand() < 0.7) {
        G4double Q2_scale = 2.0*GeV*GeV;
        Q2 = -Q2_scale * log(G4UniformRand());
    } else {
        Q2 = 0.5*GeV*GeV + 2.5*GeV*GeV * G4UniformRand();
    }
    
    G4double Q2_max = 2.0*M_N*Enu;
    if (Q2 > Q2_max) Q2 = Q2_max * 0.8;
    
    if (Q2 < 0.1*GeV*GeV) Q2 = 0.1*GeV*GeV;
    
    
    G4double W2 = M_N*M_N + 2.0*M_N*nu - Q2;
    G4double W;
    
    if (W2 > 0) {
        W = sqrt(W2);
        if (W < 1.1*GeV) W = 1.1*GeV + 0.5*GeV * G4UniformRand();
    } else {
        W = 1.1*GeV + 0.3*GeV * G4UniformRand();
        Q2 = M_N*M_N + 2.0*M_N*nu - W*W;
        if (Q2 < 0.1*GeV*GeV) Q2 = 0.1*GeV*GeV;
    }
    
    G4double cos_theta = 1.0 - (Q2 + m_mu*m_mu)/(2.0*Enu*Emu);
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;
    
    G4double theta_mu = acos(cos_theta);
    G4double phi_mu = G4UniformRand() * 2.0 * pi;
    
    G4ThreeVector mu_dir(sin(theta_mu)*cos(phi_mu), 
                         sin(theta_mu)*sin(phi_mu), 
                         cos(theta_mu));
    
    G4double KEmu = Emu - m_mu;
    if (KEmu <= 0) KEmu = 100*MeV;
    
    fParticleGun->SetParticleDefinition(G4MuonMinus::MuonMinus());
    fParticleGun->SetParticleEnergy(KEmu);
    fParticleGun->SetParticleMomentumDirection(mu_dir);
    fParticleGun->SetParticlePosition(position);
    fParticleGun->GeneratePrimaryVertex(anEvent);
    
    G4double p_mu_sq = Emu*Emu - m_mu*m_mu;
    G4double p_mu;
    if (p_mu_sq > 0) {
        p_mu = sqrt(p_mu_sq);
    } else {
        p_mu = 0.05*GeV;  
    }
    G4ThreeVector p_mu_vec = mu_dir * p_mu;
    
    G4double E_available = Enu - Emu;
    G4int nParticles = 5 + G4int(G4UniformRand() * 5);
    
    std::vector<G4ParticleDefinition*> hadrons = {
        G4Proton::Proton(), G4Neutron::Neutron(),
        G4PionPlus::PionPlus(), G4PionMinus::PionMinus(), 
        G4PionZero::PionZero()
    };
    
    G4double E_per = E_available / nParticles;
    G4double E_rem = E_available;
    
    for (G4int i = 0; i < nParticles; i++) {
        G4double frac = exp(-2.5 * G4UniformRand());
        G4double E_had = E_per * frac * 2.5;
        
        if (E_had < 70*MeV) E_had = 70*MeV;
        if (E_had > E_rem * 0.7) E_had = E_rem * 0.7;
        if (i == nParticles - 1) E_had = E_rem * 0.8;
        
        E_rem -= E_had;
        if (E_rem < 0) E_rem = 0;
        
        if (E_had > 60*MeV) {
            G4ParticleDefinition* hadron = hadrons[G4int(G4UniformRand() * hadrons.size())];
            fParticleGun->SetParticleDefinition(hadron);
            fParticleGun->SetParticleEnergy(E_had);
            fParticleGun->SetParticleMomentumDirection(G4RandomDirection());
            fParticleGun->SetParticlePosition(position);
            fParticleGun->GeneratePrimaryVertex(anEvent);
        }
    }

    eventInfo->SetPrimaryVertex(position);
    eventInfo->SetPrimaryMomentum(p_mu_vec/GeV);  
    eventInfo->SetPrimaryEnergy(Emu / GeV);
    eventInfo->SetPrimaryPDG(13);  
    
    eventInfo->SetNeutrinoKinematics(Q2/(GeV*GeV), W/GeV, Enu/GeV, "DIS");
}

void PrimaryGeneratorAction::GenerateTestParticle(G4Event* anEvent)
{
    fParticleGun->SetParticleDefinition(G4MuonMinus::MuonMinus());
    fParticleGun->SetParticleEnergy(2.*GeV);
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0,0,-1));
    fParticleGun->SetParticlePosition(G4ThreeVector(0,0,2.*m));
    
    fParticleGun->GeneratePrimaryVertex(anEvent);
}

G4double PrimaryGeneratorAction::SampleCosmicEnergy()
{
    G4double alpha = 2.7;
    G4double rand = G4UniformRand();
    G4double energy = fCosmicMinEnergy * pow(
        (pow(fCosmicMaxEnergy/fCosmicMinEnergy, 1-alpha) - 1) * rand + 1,
        1./(1-alpha)
    );
    
    if (energy < 0.5*GeV) {
        energy = 0.5*GeV + G4UniformRand() * 0.5*GeV;
    }
    
    return energy;
}

G4double PrimaryGeneratorAction::SampleCosmicAngle()
{
    G4double cosMinTheta = cos(fCosmicMaxTheta);
    G4double cosMaxTheta = cos(fCosmicMinTheta);
    
    G4double randCos = cosMinTheta + G4UniformRand() * (cosMaxTheta - cosMinTheta);
    G4double cosTheta = sqrt(randCos);
    
    return acos(cosTheta);
}

G4ThreeVector PrimaryGeneratorAction::SampleCosmicPosition()
{
    G4double height = 3.*m;
    G4double spread = 4.*m;
    
    G4double x = (G4UniformRand() - 0.5) * spread;
    G4double y = (G4UniformRand() - 0.5) * spread;
    G4double z = height;
    
    return G4ThreeVector(x, y, z);
}

G4ThreeVector PrimaryGeneratorAction::FindLArEntryPoint(
    G4ThreeVector position, 
    G4ThreeVector direction)
{
    G4double halfX = DetectorConstruction::fTPC_X / 2.0 * cm;
    G4double halfY = DetectorConstruction::fTPC_Y / 2.0 * cm;
    G4double halfZ = DetectorConstruction::fTPC_Z / 2.0 * cm;
    
    G4ThreeVector entryPoint = position;
    G4double tMin = 1e10;
    G4bool foundIntersection = false;
    
    std::vector<std::pair<G4int, G4double>> faces = {
        {0, -halfX}, {0, halfX},
        {1, -halfY}, {1, halfY},
        {2, -halfZ}, {2, halfZ}
    };
    
    for (const auto& face : faces) {
        G4int axis = face.first;
        G4double boundary = face.second;
        
        if (std::abs(direction[axis]) < 1e-6) continue;
        
        G4double t = (boundary - position[axis]) / direction[axis];
        
        if (t < 0 || t > tMin) continue;
        
        G4ThreeVector intersection = position + t * direction;
        
        bool withinBounds = true;
        
        if (axis != 0) {
            if (std::abs(intersection.x()) > halfX) withinBounds = false;
        }
        if (axis != 1) {
            if (std::abs(intersection.y()) > halfY) withinBounds = false;
        }
        if (axis != 2) {
            if (std::abs(intersection.z()) > halfZ) withinBounds = false;
        }
        
        if (withinBounds) {
            tMin = t;
            entryPoint = intersection;
            foundIntersection = true;
        }
    }
    
    
    if (!foundIntersection) {
        entryPoint = G4ThreeVector(
            (G4UniformRand() - 0.5) * 2.0 * halfX,
            (G4UniformRand() - 0.5) * 2.0 * halfY,
            halfZ  
        );
    }
    
    
    if (std::abs(entryPoint.x()) > halfX + 0.1*cm ||
        std::abs(entryPoint.y()) > halfY + 0.1*cm ||
        std::abs(entryPoint.z()) > halfZ + 0.1*cm) {
        entryPoint.setZ(halfZ);
        if (std::abs(entryPoint.x()) > halfX) entryPoint.setX(halfX * (entryPoint.x() > 0 ? 1 : -1));
        if (std::abs(entryPoint.y()) > halfY) entryPoint.setY(halfY * (entryPoint.y() > 0 ? 1 : -1));
    }
    
    return entryPoint;
}

void PrimaryGeneratorAction::SetCosmicEnergyRange(G4double minE, G4double maxE)
{
    fCosmicMinEnergy = minE;
    fCosmicMaxEnergy = maxE;
}

void PrimaryGeneratorAction::SetCosmicAngleRange(G4double minTheta, G4double maxTheta)
{
    fCosmicMinTheta = minTheta;
    fCosmicMaxTheta = maxTheta;
}

void PrimaryGeneratorAction::SetNeutrinoEnergyRange(G4double minE, G4double maxE)
{
    fNeutrinoMinEnergy = minE;
    fNeutrinoMaxEnergy = maxE;
}

void PrimaryGeneratorAction::InitializeBNBFlux()
{
    G4cout << "========================================" << G4endl;
    G4cout << "Initializing BNB Flux" << G4endl;

    for (G4int i = 0; i <= fNBNBBins; i++) {
        fBNBEnergyBins[i] = i * 0.05;
    }
    
    G4double fluxData[60] = {
        3.09,11.9,15.3,18.3,22.7,25.0,26.7,28.0,29.6,30.9,31.6,31.6,31.2,30.9,30.6,
        29.9,28.7,27.5,26.3,24.9,23.6,22.2,20.6,19.2,17.8,16.2,14.7,13.2,11.7,10.2,
        8.85,7.65,6.50,5.48,4.63,3.83,3.18,2.57,2.10,1.70,1.35,1.11,0.911,0.723,
        0.621,0.535,0.463,0.404,0.367,0.333,0.308,0.289,0.278,0.269,0.258,0.240,
        0.237,0.229,0.225,0.209
    };
    
    G4double totalFlux = 0.0;
    for (G4int i = 0; i < fNBNBBins; i++) {
        fBNBFluxWeights[i] = fluxData[i];
        totalFlux += fluxData[i];
    }
    
    fBNBCumulativeWeights[0] = fBNBFluxWeights[0] / totalFlux;
    for (G4int i = 1; i < fNBNBBins; i++) {
        fBNBCumulativeWeights[i] = fBNBCumulativeWeights[i-1] + fBNBFluxWeights[i] / totalFlux;
    }
    
    G4cout << "BNB Flux initialized: 60 bins, 0-3 GeV" << G4endl;
    G4cout << "Peak flux at: 0.65-0.70 GeV (31.6×10⁻¹¹ νμ/POT/cm²)" << G4endl;
    G4cout << "========================================" << G4endl;
}

G4double PrimaryGeneratorAction::SampleBNBEnergy()
{
    G4double rand = G4UniformRand();
    
    G4int binIndex = 0;
    for (G4int i = 0; i < fNBNBBins; i++) {
        if (rand < fBNBCumulativeWeights[i]) {
            binIndex = i;
            break;
        }
    }
    
    G4double Emin = fBNBEnergyBins[binIndex];
    G4double Emax = fBNBEnergyBins[binIndex + 1];
    G4double energy = Emin + G4UniformRand() * (Emax - Emin);
    
    return energy * GeV;
}

G4double PrimaryGeneratorAction::SampleCCQE_Q2(G4double Enu)
{
    const G4double MA = 0.99*GeV;
    const G4double MA2 = MA*MA;
    
    G4double Q2_max = std::min(2.0*GeV*GeV, 2.0*0.939*GeV*Enu);
    
    G4double u = G4UniformRand();
    G4double Q2 = MA2 * u / (1.0 - 0.9*u);
    
    if (Q2 > Q2_max) Q2 = Q2_max * G4UniformRand();
    if (Q2 < 0.02*GeV*GeV) Q2 = 0.02*GeV*GeV;
    if (Q2 > 1.5*GeV*GeV) Q2 = 0.5*GeV*GeV * G4UniformRand();
    
    return Q2;
}

G4double PrimaryGeneratorAction::SampleResonance_W()
{
    const G4double M_Delta = 1.232*GeV;
    const G4double Gamma_Delta = 0.120*GeV;

    G4double W;
    G4bool accept = false;
    G4int maxTries = 1000;
    G4int tries = 0;
    
    while (!accept && tries < maxTries) {
        W = M_Delta + (G4UniformRand() - 0.5) * 0.8*GeV;
        
        if (W < 1.08*GeV || W > 2.0*GeV) {
            tries++;
            continue;
        }
        
        G4double W2 = W*W;
        G4double M2 = M_Delta*M_Delta;
        G4double MG = M_Delta * Gamma_Delta;
        G4double BW = Gamma_Delta / (pow(W2 - M2, 2) + MG*MG);
        
        G4double normalized = BW * MG * MG;
        
        if (G4UniformRand() < normalized) accept = true;
        tries++;
    }
    
    if (tries >= maxTries) W = 1.232*GeV;
    
    return W;
}

G4double PrimaryGeneratorAction::SampleDIS_y(G4double Enu)
{
    G4double y;
    G4bool accept = false;
    G4int maxTries = 1000;
    G4int tries = 0;
    
    while (!accept && tries < maxTries) {
        y = G4UniformRand();
        
        G4double weight = (1.0 + pow(1.0 - y, 2)) / 2.0;
        
        if (G4UniformRand() < weight) accept = true;
        tries++;
    }
    
    if (tries >= maxTries) y = 0.3;
    
    if (y < 0.1) y = 0.1;
    if (y > 0.9) y = 0.9;
    
    return y;
}
