#include "DetectorConstruction.hh"
#include "LArSensitiveDetector.hh"

#include "G4RunManager.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4SDManager.hh"

DetectorConstruction::DetectorConstruction()
    : G4VUserDetectorConstruction(),
      fLAr(nullptr), fAir(nullptr), fSteel(nullptr),
      fWorldLogical(nullptr), fLArLogical(nullptr), fCryostatLogical(nullptr),
      fWorldPhysical(nullptr), fLArPhysical(nullptr)
{
    DefineMaterials();
}

DetectorConstruction::~DetectorConstruction()
{}

void DetectorConstruction::DefineMaterials()
{
    G4NistManager* nist = G4NistManager::Instance();
    
    G4Element* Ar = nist->FindOrBuildElement("Ar");
    fLAr = new G4Material("LiquidArgon", 1.396*g/cm3, 1, kStateLiquid, 87.*kelvin);
    fLAr->AddElement(Ar, 1);
    
    const G4int nEntries = 32;
    G4double photonEnergy[nEntries] = {
        2.034*eV, 2.068*eV, 2.103*eV, 2.139*eV, 2.177*eV, 2.216*eV, 2.256*eV, 2.298*eV,
        2.341*eV, 2.386*eV, 2.433*eV, 2.481*eV, 2.532*eV, 2.585*eV, 2.640*eV, 2.697*eV,
        2.757*eV, 2.820*eV, 2.885*eV, 2.954*eV, 3.026*eV, 3.102*eV, 3.181*eV, 3.265*eV,
        3.353*eV, 3.446*eV, 3.545*eV, 3.649*eV, 3.760*eV, 3.877*eV, 4.002*eV, 4.136*eV
    };
    
    G4double refractiveIndex[nEntries];
    for (int i = 0; i < nEntries; i++) {
        refractiveIndex[i] = 1.38;
    }

    G4double absorption[nEntries];
    for (int i = 0; i < nEntries; i++) {
        absorption[i] = 100.*m;
    }

    G4double scintillation[nEntries];
    for (int i = 0; i < nEntries; i++) {
        scintillation[i] = 1.0;
    }
    
    G4double rayleigh[nEntries];
    for (int i = 0; i < nEntries; i++) {
        rayleigh[i] = 90.0*cm;
    }

    G4MaterialPropertiesTable* larMPT = new G4MaterialPropertiesTable();

    larMPT->AddProperty("RINDEX", photonEnergy, refractiveIndex, nEntries);
    larMPT->AddProperty("ABSLENGTH", photonEnergy, absorption, nEntries);
    larMPT->AddProperty("RAYLEIGH", photonEnergy, rayleigh, nEntries);
    larMPT->AddProperty("SCINTILLATIONCOMPONENT1", photonEnergy, scintillation, nEntries);
    larMPT->AddProperty("SCINTILLATIONCOMPONENT2", photonEnergy, scintillation, nEntries);

    larMPT->AddConstProperty("SCINTILLATIONYIELD", 24000.0/MeV, true);
    larMPT->AddConstProperty("RESOLUTIONSCALE", 1.0, true);
    larMPT->AddConstProperty("SCINTILLATIONTIMECONSTANT1", 6.0*ns, true);
    larMPT->AddConstProperty("SCINTILLATIONTIMECONSTANT2", 1590.0*ns, true);
    larMPT->AddConstProperty("SCINTILLATIONYIELD1", 0.23, true);
    larMPT->AddConstProperty("SCINTILLATIONYIELD2", 0.77, true);

    fLAr->SetMaterialPropertiesTable(larMPT);
    
    G4cout << "========================================" << G4endl;
    G4cout << "Liquid Argon Optical Properties (Geant4 11.x):" << G4endl;
    G4cout << "  Scintillation Yield: 24,000 photons/MeV" << G4endl;
    G4cout << "  Component 1 (Fast): 6 ns decay, 23% yield" << G4endl;
    G4cout << "  Component 2 (Slow): 1590 ns decay, 77% yield" << G4endl;
    G4cout << "  Refractive index: 1.38" << G4endl;
    G4cout << "  Absorption length: 100 m" << G4endl;
    G4cout << "  Rayleigh scattering: 90 cm" << G4endl;
    G4cout << "========================================" << G4endl;

    fAir = nist->FindOrBuildMaterial("G4_AIR");
    fSteel = nist->FindOrBuildMaterial("G4_STAINLESS-STEEL");

    G4cout << "Materials defined successfully" << G4endl;
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    G4double worldSize = 6.*m;
    G4Box* solidWorld = new G4Box("World", 
        0.5*worldSize, 0.5*worldSize, 0.5*worldSize);
    
    fWorldLogical = new G4LogicalVolume(solidWorld, fAir, "World");
    
    fWorldPhysical = new G4PVPlacement(0,
                                       G4ThreeVector(),
                                       fWorldLogical,
                                       "World",
                                       0,
                                       false,
                                       0,
                                       true);

    G4double cryostatThickness = 10.*cm;
    G4Box* solidCryostat = new G4Box("Cryostat",
        0.5*(fTPC_X*cm + 2*cryostatThickness),
        0.5*(fTPC_Y*cm + 2*cryostatThickness),
        0.5*(fTPC_Z*cm + 2*cryostatThickness));
    
    fCryostatLogical = new G4LogicalVolume(solidCryostat, fSteel, "Cryostat");
    
    new G4PVPlacement(0,
                      G4ThreeVector(),
                      fCryostatLogical,
                      "Cryostat",
                      fWorldLogical,
                      false,
                      0,
                      true);

    G4Box* solidLAr = new G4Box("LArTPC",
        0.5*fTPC_X*cm, 0.5*fTPC_Y*cm, 0.5*fTPC_Z*cm);
    
    fLArLogical = new G4LogicalVolume(solidLAr, fLAr, "LArTPC");
    
    fLArPhysical = new G4PVPlacement(0,
                                     G4ThreeVector(),
                                     fLArLogical,
                                     "LArTPC",
                                     fCryostatLogical,
                                     false,
                                     0,
                                     true);

    G4VisAttributes* worldVisAtt = new G4VisAttributes(G4Colour(1.0, 1.0, 1.0, 0.0));
    worldVisAtt->SetVisibility(false);
    fWorldLogical->SetVisAttributes(worldVisAtt);
    
    G4VisAttributes* cryostatVisAtt = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 0.3));
    fCryostatLogical->SetVisAttributes(cryostatVisAtt);
    
    G4VisAttributes* larVisAtt = new G4VisAttributes(G4Colour(0.0, 0.8, 1.0, 0.3));
    larVisAtt->SetVisibility(true);
    fLArLogical->SetVisAttributes(larVisAtt);
    
    G4cout << "Geometry constructed successfully" << G4endl;
    G4cout << "TPC dimensions: " << fTPC_X << " x " << fTPC_Y << " x " << fTPC_Z << " cm^3" << G4endl;
    
    return fWorldPhysical;
}

void DetectorConstruction::ConstructSDandField()
{
    G4String sdName = "LArTPC/LArSD";
    LArSensitiveDetector* larSD = new LArSensitiveDetector(sdName, "LArHitsCollection");
    
    G4SDManager::GetSDMpointer()->AddNewDetector(larSD);
    
    fLArLogical->SetSensitiveDetector(larSD);
    
    G4cout << "Sensitive detector attached to LAr volume only" << G4endl;
}
