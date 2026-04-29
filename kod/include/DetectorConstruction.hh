#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Material.hh"
#include "G4VisAttributes.hh"
#include "globals.hh"

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    virtual ~DetectorConstruction();

    virtual G4VPhysicalVolume* Construct();
    virtual void ConstructSDandField();

    G4LogicalVolume* GetLArVolume() const { return fLArLogical; }
    
    
    static constexpr G4double fTPC_X = 256.35;  
    static constexpr G4double fTPC_Y = 233.0;   
    static constexpr G4double fTPC_Z = 1036.8;  
    
    static constexpr G4double fWireSpacing = 0.3;      
    static constexpr G4double fDriftVelocity = 0.16;   
    
    static constexpr G4double fWirePlaneU_Z = -0.6;  
    static constexpr G4double fWirePlaneV_Z = 0.0;   
    static constexpr G4double fWirePlaneY_Z = 0.6;   

private:
    void DefineMaterials();
    
    G4Material* fLAr;
    G4Material* fAir;
    G4Material* fSteel;
    
    G4LogicalVolume* fWorldLogical;
    G4LogicalVolume* fLArLogical;
    G4LogicalVolume* fCryostatLogical;
    
    G4VPhysicalVolume* fWorldPhysical;
    G4VPhysicalVolume* fLArPhysical;
};

#endif