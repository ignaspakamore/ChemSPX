#!/usr/bin/python
import PySimpleGUI as sg
import _thread 
from ChemSPX.main import Program



class CSPXGUI():
    def __init__(self):
        self.indict = None
        self.stop = False


    def _form_input(self, values):
        _input = {
    'in_file':None,
    'init_data_sampling':None,
    'n_processes':None,
    'sample_number':None,
    'OPT_method':None,
    'omptimisation_cycles':None,
    


    }
        #for key, value in values.items():


    def run(self):

        sg.theme('DefaultNoMoreNagging')   # Add a touch of color

        col1 = [[sg.Frame(layout=[
                  [sg.Text('Data file'), sg.In(size=(30,1), enable_events=True), sg.FileBrowse()],
                  ], title="Input",
                    relief=sg.RELIEF_GROOVE,
                  )],
                  [sg.Frame(layout=[
                  [sg.Text('Initial sampling', justification='left', size=(10, 1)), sg.InputText(default_text='LHS', size=(10, 1))],
                  [sg.Text('n processes', justification='left', size=(10, 1)), sg.InputText(default_text='1', size=(10, 1))],
                  [sg.Text('Sample num.', justification='left', size=(10, 1)), sg.InputText(size=(10, 1))],
                  [sg.Text('Optimiser', justification='left', size=(10, 1)), sg.InputText(default_text='BO', size=(10, 1))],
                  [sg.Text('Opt. Cycles', justification='left', size=(10, 1)), sg.InputText(default_text='5', size=(10, 1))],
                  [sg.Text('Itt. number', justification='left', size=(10, 1)), sg.InputText(default_text='120', size=(10, 1))],
                  [sg.Text('Method', justification='left', size=(10, 1)), sg.InputText(default_text='full_space', size=(10, 1))],
                  ], title="OPTIMISATION",
                    relief=sg.RELIEF_GROOVE,
                  )],
                  [sg.Frame(layout=[
                  [sg.Text('f(x)', justification='left', size=(10, 1)), sg.InputText(default_text='Force', size=(10, 1))],
                  [sg.Text('Xi', justification='left', size=(10, 1)), sg.InputText(default_text='0.01', size=(10, 1))],
                  [sg.Text('power', justification='left', size=(10, 1)), sg.InputText(default_text='1', size=(10, 1))],
                  [sg.Text('NN', justification='left', size=(10, 1)), sg.InputText(default_text='all', size=(10, 1))],
                  ], title="FUNCTION",
                    relief=sg.RELIEF_GROOVE,
                  )],
                  [sg.Frame(layout=[
                  [sg.Text('f(x)', justification='left', size=(10, 1)), sg.InputText(default_text='0', size=(10, 1))], 
                  [sg.Text('del f(x)', justification='left', size=(10, 1)), sg.InputText(default_text='0', size=(10, 1))], 
                  [sg.Text('vector', justification='left', size=(10, 1)), sg.InputText(default_text='0', size=(10, 1))], 
                  ], title="CONVERGENCE",
                    relief=sg.RELIEF_GROOVE,
                  )],
                  [sg.Frame(layout=[
                  [sg.Text('Apply BD', justification='left', size=(10, 1)), sg.InputText(default_text='True', size=(10, 1))],
                  [sg.Text('UBL', justification='left', size=(10, 1)), sg.InputText(size=(20, 1))],
                  [sg.Text('LBL', justification='left', size=(10, 1)), sg.InputText(size=(20, 1))],
                  ], title="BOUNDARY CONDITIONS",
                    relief=sg.RELIEF_GROOVE,
                  )],
                  [sg.Button('Generate input'),sg.Button('Edit input'), sg.Button('RUN')]]

        output = sg.Text()
        col2 = [[output]]

        # All the stuff inside your window.

        layout = [[sg.Column(col1, element_justification='left'), sg.Column(col2, element_justification='right')]]

        # Create the Window
        window = sg.Window('CSPX-GUI', layout, size=(1300, 700), font=30)
        # Event Loop to process "events" and get the "values" of the inputs
        while True:
            event, values = window.read()
            if event == 'RUN':
                #program = Program(values).run()
                pass
                

            if event == 'STOP':
                pass

            elif event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
                break
                


        window.close()


if __name__ == "__main__":
    program = CSPXGUI()

    program.run()



