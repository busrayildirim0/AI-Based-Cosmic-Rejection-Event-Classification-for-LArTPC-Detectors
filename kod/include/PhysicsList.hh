#ifndef PhysicsList_h
#define PhysicsList_h 1

#include "G4VModularPhysicsList.hh"
#include "globals.hh"

class G4OpticalPhysics;

class PhysicsList : public G4VModularPhysicsList
{
public:
    PhysicsList();
    virtual ~PhysicsList();

    virtual void SetCuts();

private:
    G4OpticalPhysics* fOpticalPhysics;
    G4int fVerboseLevel;
    
    void ConfigureOpticalPhysics();
};

#endif
