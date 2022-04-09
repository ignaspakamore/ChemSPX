import
import PySimpleGUI as sg


class CSPXGUI():
	def __init__(self):
		self.indict = None

	def run(self):

		sg.theme('Default')   # Add a touch of color
		

		# ------ Menu Definition ------ #
		menu_def = [['File', ['Open', 'Save',]],
            ['Edit', ['Paste', ['Special', 'Normal',], 'Undo'],],
            ['Help', 'About...'],]

		# All the stuff inside your window.
		layout = [[sg.Menu(menu_def)],
					[sg.Text('Data file'), sg.InputText()],
					[sg.Text('Number of components')],
					[sg.Checkbox('2',size = (4,2)), sg.Checkbox('3',size = (4,2)), sg.Text('other:'), sg.InputText()],
					[sg.Button('RUN'),sg.Button('Generate input'), sg.Button('')],
					[sg.Txt('', size=(8,1), key='output')  ] ]

		# Create the Window
		window = sg.Window('CSPX-GUI', layout, size=(350, 150), font=30)
		# Event Loop to process "events" and get the "values" of the inputs
		while True:
			event, values = window.read()

			if event == 'Calculate':
				'''
				Set calculation parameters
				'''
				self.data = values[0]
				if values[1] == True:
					self.n_componenets = 2
				elif values[2] == True:
					self.n_componenets = 3
				elif values[3] != '':
					self.n_componenets = int(values[3])

			form.FindElement('output').Update('Finished')
				

			elif event == 'PLOT':
				pass
			elif event == 'SAVE':
				pass
			elif event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
				break

		window.close()