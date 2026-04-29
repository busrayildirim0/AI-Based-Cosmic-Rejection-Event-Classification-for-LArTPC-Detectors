// PrimaryGeneratorMessenger.hh
#ifndef PrimaryGeneratorMessenger_h
#define PrimaryGeneratorMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

class PrimaryGeneratorAction;
class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWithAString;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;

class PrimaryGeneratorMessenger: public G4UImessenger
{
public:
    PrimaryGeneratorMessenger(PrimaryGeneratorAction*);
    virtual ~PrimaryGeneratorMessenger();
    
    virtual void SetNewValue(G4UIcommand*, G4String);
    
private:
    PrimaryGeneratorAction* fPrimaryGenerator;
    
    G4UIdirectory*             fGeneratorDir;
    G4UIcmdWithAString*        fSetModeCmd;
    G4UIcommand*               fSetNeutrinoEnergyCmd;
    G4UIcommand*               fSetCosmicEnergyCmd;
    G4UIcommand*               fSetCosmicAngleCmd;
};

#endif
