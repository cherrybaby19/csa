import os
import argparse

MemSize = 1000 # memory size, in reality, the memory size should be 2^32, but for this lab, for the space resaon, we keep it as this large number, but the memory is still 32-bit addressable.
def sign(num):
    if num[0] == "0":
        return int(num, 2)
    return -(-int(num, 2) & (2 ** len(num) - 1))

class InsMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir

        with open(ioDir + "/input/testcase0/imem.txt") as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def readInstr(self, ReadAddress):
        #read instruction memory
        #return 32 bit hex val

        address = int(ReadAddress)
        instruction = ''

        for i in range(0,4):
            # Fetches the instruction from the memomry 

            byte = self.IMem[address + i]
            instruction = instruction + byte

        # instruction is converted to hex 
        return hex(int(instruction, 2))

    def instrFetch(self, instruction):
        bin_instruction = (bin(int(instruction, 16)).replace('0b', '')).rjust(32, '0')

        #Segementation of instruction for easier analysis
        
        bits_0_to_6   = bin_instruction[-7:]
        bits_7_to_11  = bin_instruction[-12:-7]
        bits_12_to_14 = bin_instruction[-15:-12]
        bits_15_to_19 = bin_instruction[-20:-15]
        bits_20_to_24 = bin_instruction[-25:-20]
        bits_25_to_31 = bin_instruction[-32:-25]

        opcode = bits_0_to_6

        args = {}

        cmd = ''
        #HALT
        if opcode in ['1111111']:
            args['format'] = 'HALT'
            args['instr'] = 'HALT'

            cmd = "HALT"

        #R-Type
        elif opcode in ["0110011"]:

            rd     = bits_7_to_11
            funct3 = bits_12_to_14
            rs1    = bits_15_to_19
            rs2    = bits_20_to_24
            funct7 = bits_25_to_31

            #ADD Instr
            if funct3 == '000' and funct7 == '0000000':
                instr = 'ADD'

            #SUB Instr
            elif funct3 == '000' and funct7 == '0100000':
                instr = 'SUB'

            #SLL Instr
            elif funct3 == '001' and funct7 == '0000000':
                instr = 'SLL'

            #SLT Instr
            elif funct3 == '010' and funct7 == '0000000':
                instr = 'SLT'

            #SLTU Instr
            elif funct3 == '011' and funct7 == '0000000':
                instr = 'SLTU'

            #XOR Instr
            elif funct3 == '100' and funct7 == '0000000':
                instr = 'XOR'

            #SRL Instr
            elif funct3 == '101' and funct7 == '0000000':
                instr = 'SRL'

            #SRA Instr
            elif funct3 == '101' and funct7 == '0100000':
                instr = 'SRA'

            #OR Instr
            elif funct3 == '110' and funct7 == '0000000':
                instr = 'OR'

            #AND Instr
            elif funct3 == '111' and funct7 == '0000000':
                instr = 'AND'

            args['format'] = 'R'
            args['instr'] = instr
            args['rd']     = int(rd, 2)
            args['funct3'] = funct3
            args['rs1']    = int(rs1, 2)
            args['rs2']    = int(rs2, 2)
            args['funct7'] = funct7
            args['opcode'] = opcode

            cmd = f"{args['instr']} x{args['rd']}, x{args['rs1']}, x{args['rs2']}"

        #I-Type
        elif opcode in ["0010011", "0000011", "1100111"]:

            rd        = bits_7_to_11
            funct3    = bits_12_to_14
            rs1       = bits_15_to_19
            imm_11__0 = bits_25_to_31 + bits_20_to_24

            #For the Arithmetic Instructions
            if opcode == '0010011':

                #ADDI Instr
                if funct3 == '000':
                    instr = 'ADDI'

                #SLTI Instr
                elif funct3 == '010':
                    instr = 'SLTI'

                #SLTIU Instr
                elif funct3 == '011':
                    instr = 'SLTIU'

                #XORI Instr
                elif funct3 == '100':
                    instr = 'XORI'

                #ORI Instr
                elif funct3 == '110':
                    instr = 'ORI'

                #ANDI Instr
                elif funct3 == '111':
                    instr = 'ANDI'

                #SLLI Instr
                elif funct3 == '001':
                    instr = 'SLLI'

                #SRLI Instr
                elif funct3 == '101' and imm_11__0 == '0':
                    instr = 'SRLI'

                #SRAI Instr
                elif funct3 == '101' and imm_11__0 == '1':
                    instr = 'SRAI'

            #For the Load Instructions
            elif opcode == '0000011':

                #LB Instr
                if funct3 == '000':
                    instr = 'LB'

                #LH Instr
                elif funct3 == '001':
                    instr = 'LH'

                #LW Instr
                elif funct3 == '010':
                    instr = 'LW'

                #LBU Instr
                elif funct3 == '100':
                    instr = 'LBU'

                #LHU Instr
                elif funct3 == '101':
                    instr = 'LHU'

            #JALR Instr
            if opcode == '1100111':
                instr = 'JALR'

            if imm_11__0[0] == '1':
                imm = -int(''.join(['1' if c == '0' else '0' for c in imm_11__0]), 2) - 1
            else:
                imm =  int(imm_11__0, 2)

            args['format']   = 'I'
            args['instr']   = instr
            args['opcode']   = opcode
            args['rd']       = int(rd, 2)
            args['funct3']   = funct3
            args['rs1']      = int(rs1, 2)
            args['imm 11:0'] = int(imm_11__0, 2)
            args['imm']      = imm

            if opcode == "0000011":
                cmd = f"{args['instr']} x{args['rd']}, {args['imm']}(x{args['rs1']})"

            else:
                cmd = f"{args['instr']} x{args['rd']}, x{args['rs1']}, {args['imm']}"

        #S-Type
        elif opcode in ["0100011"]:

            imm_4__0  = bits_7_to_11
            funct3    = bits_12_to_14
            rs1       = bits_15_to_19
            rs2       = bits_20_to_24
            imm_11__5 = bits_25_to_31

            #SB Instr
            if funct3 == '000':
                instr = 'SB'

            #SH Instr
            elif funct3 == '001':
                instr = 'SH'

            #SW Instr
            elif funct3 == '010':
                instr = 'SW'

            imm = imm_11__5 + imm_4__0
            if imm[0] == '1':
                imm = -int(''.join(['1' if c == '0' else '0' for c in imm]), 2) - 1
            else:
                imm =  int(imm, 2)


            args['format']   = 'S'
            args['instr']   = instr
            args['opcode']   = opcode
            args['imm 4:0']  = int(imm_4__0, 2)
            args['funct3']   = funct3
            args['rs1']      = int(rs1, 2)
            args['rs2']      = int(rs2, 2)
            args['imm 11:5'] = int(imm_11__5, 2)
            args['imm']      = imm

            cmd = f"{args['instr']} x{args['rs2']}, {args['imm']}(x{args['rs1']})"


        #SB-Type
        elif opcode in ["1100011"]:

            imm_4__1_11  = bits_7_to_11
            funct3       = bits_12_to_14
            rs1          = bits_15_to_19
            rs2          = bits_20_to_24
            imm_12_10__5 = bits_25_to_31

            #BEQ Instr
            if funct3 == '000':
                instr = 'BEQ'

            #BNE Instr
            elif funct3 == '001':
                instr = 'BNE'
 
            #BLT Instr
            elif funct3 == '100':
                instr = 'BLT'

            #BGE Instr
            elif funct3 == '101':
                instr = 'BGE'

            #BLTU Instr
            elif funct3 == '110':
                instr = 'BLTU'

            #BGEU Instr
            elif funct3 == '111':
                instr = 'BGEU'

            imm = imm_12_10__5 + imm_4__1_11
            imm = "".join((bin_instruction[-32], bin_instruction[-8], bin_instruction[-31:-25], bin_instruction[-12:-8], "0",))

            if imm[0] == "0":
                imm = int("0b" + imm, 2)
            else:
                imm =  -(-int("0b" + imm, 2,)& 0b11111111111)


            args['format']      = 'SB'
            args['instr']      = instr
            args['opcode']      = opcode
            args['imm 4:1|11']  = bits_7_to_11
            args['funct3']      = bits_12_to_14
            args['rs1']         = int(bits_15_to_19, 2)
            args['rs2']         = int(bits_20_to_24, 2)
            args['imm 12|10:5'] = bits_25_to_31
            args['imm']         = imm

            cmd = f"{args['instr']} x{args['rs1']}, x{args['rs2']}, label #imm = {args['imm']}"

        #U-Type
        elif opcode in ["0110111", "0010111"]:

            rd         = bits_7_to_11
            imm_31__12 = bits_25_to_31 + bits_20_to_24 + bits_15_to_19 + bits_12_to_14

            #LUI Instr
            if opcode == '0110111':
                instr = 'LUI'

            #AUIPC Instr
            elif opcode == '0010111':
                instr = 'AUIPC'

            imm = imm_31__12
            if imm[0] == '1':
                imm = -int(''.join(['1' if c == '0' else '0' for c in imm]), 2) - 1
            else:
                imm =  int(imm, 2)

            args['format']    = 'U'
            args['instr']    = instr
            args['opcode']    = opcode
            args['rd']        = int(rd, 2)
            args['imm']       = imm

            cmd = f"{args['instr']} x{args['rd']}, {args['imm']}"

        #UJ-Type
        elif opcode in ["1101111"]:

            rd                     = bin_instruction[-12:-7]
            imm_20_10__1_11_19__12 = bits_25_to_31 + bits_20_to_24 + bits_15_to_19 + bits_12_to_14

            imm = imm_20_10__1_11_19__12
            imm = "".join((bin_instruction[-32], bin_instruction[-20:-12], bin_instruction[-21],bin_instruction[-31:-21] ))

            if imm[0] == "0":
                imm = int("0b" + imm, 2)
            else:
                imm = -(-int("0b" + imm, 2,)& 0b11111111111)

            args['format']               = 'J'
            args['instr']               = 'JAL'
            args['opcode']               = opcode
            args['rd']                   = int(bits_7_to_11, 2)
            args['imm 20|10:1|11|19:12'] = imm_20_10__1_11_19__12
            args['imm']                  = imm

            cmd = f"{args['instr']} x{args['rd']}, {args['imm']}"

        

        return(args)

class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(ioDir + "/input/testcase0/dmem.txt") as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]
            for i in range(1000 - len(self.DMem)):
                self.DMem.append("00000000")

    def readDataMem(self, ReadAddress, mode = 'word'):
        #read data memory
        #return 32 bit hex val
        try:
            address = int(ReadAddress, 16)
        except:
            address = ReadAddress
        data = ''
        if mode == 'word':
            data = self.DMem[address + 0] + self.DMem[address + 1] + self.DMem[address + 2] + self.DMem[address + 3]
        elif mode == 'half':
            data = self.DMem[address + 2] + self.DMem[address + 3]
        elif mode == 'byte':
            data = self.DMem[address + 3]

        # Convert the instruction to a 32-bit hex and return
        return int(data, 2)

    def writeDataMem(self, WriteAddress, WriteData, mode = 'word'):
        # write data into byte addressable memory
        try:
            address = int(WriteAddress, 16)
        except:
            address = WriteAddress
        data = format(WriteData & 0xFFFFFFFF, '032b')
        data = [data[24:32], data[16:24], data[8:16], data[:8]][::-1]
        if mode == 'word':
            range_low, range_high = 0, 4
        elif mode == 'half':
            range_low, range_high = 2, 4
        elif mode == 'byte':
            range_low, range_high = 3, 4

        for i in range(range_low, range_high):
            
            self.DMem[address + i] = data[i]

    def outputDataMem(self):
        resPath = self.ioDir + "//" + self.id + "_DMEMResult.txt"
        with open(resPath, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])
        
class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        self.Registers = ["0b00000000000000000000000000000000" for i in range(32)]

    def readRF(self, Reg_addr):
        # Fill in
        binary_num = self.Registers[Reg_addr]
        decimal_num = int(binary_num, 2)
        if binary_num[0] == '1':  # check sign bit
            decimal_num -= 2 ** 32  # handle negative sign

        return(decimal_num)

    def writeRF(self, Reg_addr, Wrt_reg_data):
        if Reg_addr != 0:
            bin_x = bin(Wrt_reg_data & (2**32 - 1))[2:]
            self.Registers[Reg_addr] = bin_x
            if Wrt_reg_data >= 0:
                self.Registers[Reg_addr] = "0" * (32 - len(bin_x)) + bin_x
            else:
                self.Registers[Reg_addr] = "1" * (32 - len(bin_x)) + bin_x

    def outputRF(self, cycle):
        op = ["State of RF after executing cycle:	" + str(cycle) + "\n"] #"-"*70+"\n",
        op.extend([str(val).replace('0b', '')+"\n" for val in self.Registers])
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)

class State(object):
    def __init__(self):
        self.IF = {"nop": False, "PC": 0, 'instruction_count': 0}
        self.ID = {"nop": True, "Instr": 0, "is_hazard":False}
        self.EX = {"nop": True, "Read_data1": 0, "Read_data2": 0, "Imm": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "is_I_type": False, "rd_mem": 0,
                   "wrt_mem": 0, "alu_op": 0, "wrt_enable": 0}
        self.MEM = {"nop": True, "ALUresult": 0, "Store_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "rd_mem": 0,
                   "wrt_mem": 0, "wrt_enable": 0}
        self.WB = {"nop": True, "Wrt_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "wrt_enable": 0}

class Core(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem

class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(SingleStageCore, self).__init__(ioDir + "//SS_", imem, dmem)
        self.opFilePath = ioDir + "//StateResult_SS.txt"
        self.halted = False

    def step(self):
        # Your implementation

        if self.state.IF["nop"]:
            self.nextState.IF['instruction_count'] += 1
            self.halted = True

        else:
            #IF
            self.nextState.IF['nop'] = False
            self.nextState.IF['PC'] = self.state.IF['PC']

            instruction = self.ext_imem.readInstr(self.nextState.IF['PC'])
            args = self.ext_imem.instrFetch(instruction)
            if instruction != "1"*32:
                self.nextState.IF['instruction_count'] += 1

            #R-Type
            if args['format'] == 'R':

                reg_1 = self.myRF.readRF(args['rs1'])
                reg_2 = self.myRF.readRF(args['rs2'])

                #ADD
                if args['instr'] == 'ADD':
                    write_value = reg_1 + reg_2

                #SUB
                elif args['instr'] == 'SUB':
                    write_value = reg_1 - reg_2

                #XOR
                elif args['instr'] == 'XOR':
                    write_value = reg_1 ^ reg_2

                #OR
                elif args['instr'] == 'OR':
                    write_value = reg_1 | reg_2

                #AND
                elif args['instr'] == 'AND':
                    write_value = reg_1 & reg_2

                self.myRF.writeRF(args['rd'], write_value)
                self.nextState.IF['PC']  += 4

            #I-Type
            elif args['format'] == 'I':

                reg_1 = self.myRF.readRF(args['rs1'])
                imm   = args['imm']

                #ADDI
                if args['instr'] == 'ADDI':
                    write_value = reg_1 + imm

                #XORI
                elif args['instr'] == 'XORI':
                    write_value = reg_1 ^ imm

                #ORI
                elif args['instr'] == 'ORI':
                    write_value = reg_1 | imm

                #ANDI
                elif args['instr'] == 'ANDI':
                    write_value = reg_1 & imm

                #LW
                elif args['instr'] == 'LW':
                    write_value = self.ext_dmem.readDataMem(hex(reg_1 + imm), mode = 'word')

                #LH
                elif args['instr'] == 'LH':
                    write_value = self.ext_dmem.readDataMem(hex(reg_1 + imm), mode = 'half')

                #LB
                elif args['instr'] == 'LB':
                   
                    write_value = self.ext_dmem.readDataMem(hex(reg_1 + imm), mode = 'word')

                self.myRF.writeRF(args['rd'], write_value)
                self.nextState.IF['PC']  += 4

            #S-Type
            elif args['format'] == 'S':

                reg_1 = self.myRF.readRF(args['rs1'])
                reg_2 = self.myRF.readRF(args['rs2'])
                imm   = args['imm']

                #SW
                if args['instr'] == 'SW':
                    self.ext_dmem.writeDataMem(hex(reg_1 + imm), reg_2, mode = 'word')

                #SH
                elif args['instr'] == 'SW':
                    self.ext_dmem.writeDataMem(hex(reg_1 + imm), reg_2, mode = 'half')

                #SB
                elif args['instr'] == 'SW':
                    self.ext_dmem.writeDataMem(hex(reg_1 + imm), reg_2, mode = 'byte')

                self.nextState.IF['PC'] += 4

            #SB-Type
            elif args['format'] == 'SB':

                reg_1 = self.myRF.readRF(args['rs1'])
                reg_2 = self.myRF.readRF(args['rs2'])
                imm   = args['imm']

                #BEQ
                if args['instr'] == 'BEQ':
                    if reg_1 == reg_2:
                        self.nextState.IF['PC']  += imm
                    else:
                        self.nextState.IF['PC']  += 4

                #BNE
                elif args['instr'] == 'BNE':
                    if reg_1 != reg_2:
                        self.nextState.IF['PC']  += imm
                    else:
                        self.nextState.IF['PC']  += 4

            #JAL
            elif args['instr'] == 'JAL':
                imm   = args['imm']
                self.myRF.writeRF(args['rd'], self.nextState.IF['PC'] + 4)
                self.nextState.IF['PC'] = (self.nextState.IF['PC'] + imm*2)


            #HALT
            elif args['instr'] == 'HALT':
                self.nextState.IF['nop']  = True

        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0 and cycle n+1

        self.state = self.nextState #The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.append("IF.PC: " + str(state.IF['PC']) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n")

        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

class FiveStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(ioDir + "//FS_", imem, dmem)
        self.opFilePath = ioDir + "//StateResult_FS.txt"
        self.halted = False
        self.halted_patience = 0

    def step(self):
        # Your implementation
        # --------------------- WB stage ---------------------
        
        if not self.nextState.WB["nop"]:
        
            if (self.nextState.WB["wrt_enable"]==1):
                self.myRF.writeRF(self.nextState.WB["Wrt_reg_addr"], self.nextState.WB["Wrt_data"])
            if self.nextState.MEM["nop"]:
                self.nextState.WB["nop"] = True
        else:
            if not self.nextState.MEM["nop"]:
                self.nextState.WB["nop"] = False
        
        # --------------------- MEM stage --------------------
        if not self.nextState.MEM["nop"]:
            
            self.nextState.WB["Rs"] = self.nextState.MEM["Rs"]
            self.nextState.WB["Rt"] = self.nextState.MEM["Rt"]
            self.nextState.WB["Wrt_reg_addr"] = self.nextState.MEM["Wrt_reg_addr"]
            self.nextState.WB["wrt_enable"] = self.nextState.MEM["wrt_enable"]
            
            if(self.nextState.MEM["wrt_mem"]==1):
                # writing back to datamem with store
                self.ext_dmem.writeDataMem(self.nextState.MEM["ALUresult"] , self.nextState.MEM["Store_data"])
                
            elif(self.nextState.MEM["rd_mem"]==1):
                # reading from register with load
                self.nextState.WB["Wrt_data"] = self.ext_dmem.readDataMem(self.nextState.MEM["ALUresult"])
                # 
                
            elif(self.nextState.MEM["wrt_mem"]==0 and self.nextState.MEM["rd_mem"]==0):
                # any other branch/R-type instruction does not require memory
                self.nextState.WB["Wrt_data"] = self.nextState.MEM["ALUresult"]

            if self.nextState.EX["nop"]:
                self.nextState.MEM["nop"] = True
        else:
            if not self.nextState.EX["nop"]:
                self.nextState.MEM["nop"] = False
        
        # --------------------- EX stage ---------------------
        if not self.nextState.EX["nop"]:
            
            self.nextState.MEM["wrt_enable"]=self.nextState.EX["wrt_enable"]
            self.nextState.MEM["Wrt_reg_addr"]=self.nextState.EX["Wrt_reg_addr"]
            self.nextState.MEM["rd_mem"]=self.nextState.EX["rd_mem"]
            self.nextState.MEM["wrt_mem"]=self.nextState.EX["wrt_mem"] 
            self.nextState.MEM["Rs"]=self.nextState.EX["Rs"] 
            self.nextState.MEM["Rt"]=self.nextState.EX["Rt"] 

            #R-type----------------------------------------------------------------------

            if (self.nextState.EX["alu_op"]=='add'):
                self.nextState.MEM["ALUresult"]=(self.nextState.EX["Read_data1"]) + (self.nextState.EX["Read_data2"])

            if (self.nextState.EX["alu_op"]=='sub'):
                self.nextState.MEM["ALUresult"]=self.nextState.EX["Read_data1"] - self.nextState.EX["Read_data2"]

            if (self.nextState.EX["alu_op"]=='xor'):
                self.nextState.MEM["ALUresult"]=self.nextState.EX["Read_data1"] ^ self.nextState.EX["Read_data2"]
                # 

            if (self.nextState.EX["alu_op"]=='or'):
                self.nextState.MEM["ALUresult"]=self.nextState.EX["Read_data1"] | self.nextState.EX["Read_data2"]

            if (self.nextState.EX["alu_op"]=='and'):
                self.nextState.MEM["ALUresult"]=self.nextState.EX["Read_data1"] & self.nextState.EX["Read_data2"]

            #I-type --------------------------------------------------------------------

            if (self.nextState.EX["alu_op"]=='addi'):
                self.nextState.MEM["ALUresult"]=self.nextState.EX["Read_data1"] + self.nextState.EX["Imm"]

            if (self.nextState.EX["alu_op"]=='xori'):
                self.nextState.MEM["ALUresult"]=self.nextState.EX["Read_data1"] ^ self.nextState.EX["Imm"]

            if (self.nextState.EX["alu_op"]=='ori'):
                self.nextState.MEM["ALUresult"]=self.nextState.EX["Read_data1"] | self.nextState.EX["Imm"]  

            if (self.nextState.EX["alu_op"]=='andi'):
                self.nextState.MEM["ALUresult"]=self.nextState.EX["Read_data1"] & self.nextState.EX["Imm"]  

            #Load --------------------------------------------------------------------

            if (self.nextState.EX["alu_op"]=='lw'):
                self.nextState.MEM["ALUresult"]=self.nextState.EX["Read_data1"] + self.nextState.EX["Imm"]

            #Store --------------------------------------------------------------------

            if (self.nextState.EX["alu_op"]=='sw'):
                self.nextState.MEM["ALUresult"]=self.nextState.EX["Read_data1"] + self.nextState.EX["Imm"] 
                self.nextState.MEM["Store_data"]=self.nextState.EX["Read_data2"] 

            #JAL --------------------------------------------------------------------

            if (self.nextState.EX["alu_op"]=='jal'): 
                self.nextState.MEM["Wrt_reg_addr"]=self.nextState.EX["Wrt_reg_addr"]
                self.nextState.MEM["ALUresult"] = self.nextState.EX["Read_data1"] + self.nextState.EX["Read_data2"]

                #HALT --------------------------------------------------------------------

            else:
                None 

            self.nextState.MEM["nop"] = False
            if self.nextState.ID["nop"]:
                self.nextState.EX["nop"] = True
        else:
            if not self.nextState.ID["nop"]:
                self.nextState.EX["nop"] = False
        
        
        # --------------------- ID stage ---------------------
        if not self.nextState.ID["nop"]:
            
            self.nextState.EX["nop"] = False
            self.nextState.ID["is_hazard"] = False

            if not self.nextState.EX["nop"]: 
                self.nextState.EX["Read_data1"] = 0
                self.nextState.EX["Read_data2"] = 0
                self.nextState.EX["Imm"] = 0
                self.nextState.EX["Rs"] = 0
                self.nextState.EX["Rt"] = 0
                self.nextState.EX["Wrt_reg_addr"] = 0
                self.nextState.EX["is_I_type"] = False
                self.nextState.EX["rd_mem"] = 0
                self.nextState.EX["wrt_mem"] = 0
                self.nextState.EX["alu_op"] = 0
                self.nextState.EX["wrt_enable"] = 0

            args = self.ext_imem.instrFetch(self.nextState.ID["Instr"])

            #R-type---------------------------------------------------------------------------------

            if args['format'] == 'R':  
                self.nextState.EX["Rs"] = args['rs1']
                self.nextState.EX["Rt"] = args['rs2']
                self.nextState.EX["Wrt_reg_addr"] = args['rd']

                rs1 = args['rs1']
                rs2 = args['rs2']

                hazard_rs1 = self.detectHazard(rs1)
                hazard_rs2 = self.detectHazard(rs2)

                if (hazard_rs1 == 3 or hazard_rs2 ==3):
                    pass
                else:
                    if hazard_rs1 == 1:
                        self.nextState.EX["Read_data1"] = self.nextState.MEM["ALUresult"]
                    elif hazard_rs1 == 2:
                        self.nextState.EX["Read_data1"] = self.nextState.WB["Wrt_data"]
                    else:
                        self.nextState.EX["Read_data1"] = self.myRF.readRF(rs1)

                    if hazard_rs2 == 1:
                        self.nextState.EX["Read_data2"] = self.nextState.MEM["ALUresult"]
                    elif hazard_rs2 == 2:
                        self.nextState.EX["Read_data2"] = self.nextState.WB["Wrt_data"]
                    else:
                        self.nextState.EX["Read_data2"] = self.myRF.readRF(rs2)  
                    

                    self.nextState.EX["wrt_enable"] = 1
                    self.nextState.EX["is_I_type"] = False

                    if args['instr'] == 'ADD':
                        self.nextState.EX["alu_op"] = "add"

                    elif args['instr'] == 'SUB':
                        self.nextState.EX["alu_op"] = "sub"

                    elif args['instr'] == 'XOR':
                        self.nextState.EX["alu_op"] = "xor"

                    elif args['instr'] == 'OR':
                        self.nextState.EX["alu_op"] = "or"

                    elif args['instr'] == 'AND':
                        self.nextState.EX["alu_op"] = "and"

            #I-type ---------------------------------------------------------------------------------


            elif args['format'] == 'I':  # I-type

                if args['instr'] not in ['LB', 'LH', 'LW']:
                    self.nextState.EX["Rs"] = args['rs1']
                    self.nextState.EX["Imm"] = args['imm']
                    self.nextState.EX["Wrt_reg_addr"] = args['rd']
                    
                    rs1 = args['rs1']

                    hazard_rs1 = self.detectHazard(rs1)
                    
                    if (hazard_rs1 == 3):
                        pass
                    else:
                        if hazard_rs1 == 1:
                            self.nextState.EX["Read_data1"] = self.nextState.MEM["ALUresult"]
                        elif hazard_rs1 == 2:
                            self.nextState.EX["Read_data1"] = self.nextState.WB["Wrt_data"]
                        else:
                            self.nextState.EX["Read_data1"] = self.myRF.readRF(rs1)

                        self.nextState.EX["is_I_type"] = True
                        self.nextState.EX["wrt_enable"] = 1

                        if args['instr'] == 'ADDI':
                            self.nextState.EX["alu_op"] = "addi"

                        elif args['instr'] == 'XORI':
                            self.nextState.EX["alu_op"] = "xori"

                        elif args['instr'] == 'ORI':
                            self.nextState.EX["alu_op"] = "ori"

                        elif args['instr'] == 'ANDI':
                            self.nextState.EX["alu_op"] = "andi"
                
                else:
                    self.nextState.EX["Rs"] = args['rs1']
                    self.nextState.EX["Imm"] = args['imm']
                    self.nextState.EX["Wrt_reg_addr"] = args['rd']

                    rs1 = args['rs1'] 

                    hazard_rs1 = self.detectHazard(rs1)
                    
                    if (hazard_rs1 == 3):
                        pass
                    else:
                        if hazard_rs1 == 1:
                            self.nextState.EX["Read_data1"] = self.nextState.MEM["ALUresult"]
                        elif hazard_rs1 == 2:
                            self.nextState.EX["Read_data1"] = self.nextState.WB["Wrt_data"]
                        else:
                            self.nextState.EX["Read_data1"] = self.myRF.readRF(rs1)

                        self.nextState.EX["rd_mem"] = 1
                        self.nextState.EX["wrt_enable"] = 1
                        self.nextState.EX["is_I_type"] = True
                        self.nextState.EX["alu_op"] = "lw"
                
            #Store-type ---------------------------------------------------------------------------------------

            elif args['format'] == 'S': 
                self.nextState.EX["Rs"] = args['rs1']
                self.nextState.EX["Imm"] = args['imm']
                self.nextState.EX["Rt"] = args['rs2']

                rs1 = args['rs1']
                rs2 = args['rs2']

                hazard_rs1 = self.detectHazard(rs1)
                hazard_rs2 = self.detectHazard(rs2)
                
                if (hazard_rs1 == 3 or hazard_rs2 ==3):
                    pass
                else:
                    if hazard_rs1 == 1:
                        self.nextState.EX["Read_data1"] = self.nextState.MEM["ALUresult"]
                    elif hazard_rs1 == 2:
                        self.nextState.EX["Read_data1"] = self.nextState.WB["Wrt_data"]
                    else:
                        self.nextState.EX["Read_data1"] = self.myRF.readRF(rs1)

                    if hazard_rs2 == 1:
                        self.nextState.EX["Read_data2"] = self.nextState.MEM["ALUresult"]
                    elif hazard_rs2 == 2:
                        self.nextState.EX["Read_data2"] = self.nextState.WB["Wrt_data"]
                    else:
                        self.nextState.EX["Read_data2"] = self.myRF.readRF(rs2) 

                    self.nextState.EX["is_I_type"] = True
                    self.nextState.EX["wrt_mem"] = 1
                    self.nextState.EX["alu_op"] = "sw"

            #Branch-type --------------------------------------------------------------------

            elif args['format'] == 'SB':

                self.nextState.EX["Rs"] = args['rs1']
                self.nextState.EX["Imm"] = args['imm']
                self.nextState.EX["Rt"] = args['rs2']

                rs1 = args['rs1']
                rs2 = args['rs2']

                hazard_rs1 = self.detectHazard(rs1)
                hazard_rs2 = self.detectHazard(rs2)

                if (hazard_rs1 == 3 or hazard_rs2 ==3):
                    pass
                else:
                    if hazard_rs1 == 1:
                        self.nextState.EX["Read_data1"] = self.nextState.MEM["ALUresult"]
                    elif hazard_rs1 == 2:
                        self.nextState.EX["Read_data1"] = self.nextState.WB["Wrt_data"]
                    else:
                        self.nextState.EX["Read_data1"] = self.myRF.readRF(rs1)

                    if hazard_rs2 == 1:
                        self.nextState.EX["Read_data2"] = self.nextState.MEM["ALUresult"]
                    elif hazard_rs2 == 2:
                        self.nextState.EX["Read_data2"] = self.nextState.WB["Wrt_data"]
                    else:
                        self.nextState.EX["Read_data2"] = self.myRF.readRF(rs2)        

                    if args['instr'] == 'BEQ':
                        self.nextState.EX["alu_op"] = "beq"

                    elif args['instr'] == 'BNE':
                        self.nextState.EX["alu_op"] = "bne"
                    
                    result = abs(self.nextState.EX["Read_data1"] - self.nextState.EX["Read_data2"])
                    if bool(result) == (self.nextState.EX["alu_op"] == "bne"):
                        self.nextState.IF["PC"] += self.nextState.EX["Imm"] - 4
                        self.nextState.ID["nop"] = self.nextState.EX["nop"] = True
                    else: 
                        self.nextState.EX["nop"] = True

            #JAL-type ---------------------------------------------------------------------------------------------------------
            
            elif args['instr'] == 'JAL':  
                self.nextState.EX["Imm"] = args['imm']
                self.nextState.EX["Wrt_reg_addr"] = args['rd']
                self.nextState.EX["Read_data1"] = self.nextState.IF["PC"] - 4
                self.nextState.EX["Read_data2"] = 4
                self.nextState.EX["wrt_enable"] = 1
                self.nextState.EX["alu_op"] = "jal"
                self.nextState.IF["PC"] += self.nextState.EX["Imm"]*2 - 4
                self.nextState.ID["nop"] = True
            
            else:
                self.nextState.IF["nop"] = True

            if self.nextState.EX["is_I_type"]:
                self.nextState.EX["Imm"] = self.nextState.EX["Imm"]

            if self.nextState.IF["nop"]:
                self.nextState.ID["nop"] = True
        else:
            if not self.nextState.IF["nop"]:
                self.nextState.ID["nop"] = False
        
        
        # --------------------- IF stage ---------------------
        if not self.nextState.IF["nop"]:
            if self.nextState.ID["nop"] or (self.nextState.EX["nop"] and self.nextState.ID["is_hazard"]):
                pass
            else:            
                instruction = self.ext_imem.readInstr(self.nextState.IF["PC"])
                self.nextState.IF['instruction_count'] += 1
                if instruction == "1"*32:
                    self.nextState.IF["nop"] = True
                    self.nextState.ID["nop"] = True
                else:
                    self.nextState.ID["Instr"] = instruction
                    self.nextState.IF["PC"] += 4
                    
                if not self.nextState.IF["nop"]: 
                    self.nextState.ID["nop"] = False
        
        self.halted = False
        if self.nextState.IF["nop"] and self.nextState.ID["nop"] and self.nextState.EX["nop"] and self.nextState.MEM["nop"] and self.nextState.WB["nop"]:
            self.nextState.IF['instruction_count'] += 1
            self.halted = True

        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) 
        
        self.cycle += 1

    def detectHazard(self, rs):
        
        if rs == self.nextState.MEM["Wrt_reg_addr"] and self.nextState.MEM["rd_mem"]==0:
            # EX to 1st (state.MEM["ALUresult"])
            return 1
        elif rs == self.nextState.WB["Wrt_reg_addr"] and self.nextState.WB["wrt_enable"]:
            # EX to 2nd
            # MEM to 2nd (state.WB["Wrt_data"])
            return 2
        elif rs == self.nextState.MEM["Wrt_reg_addr"] and self.nextState.MEM["rd_mem"] != 0:
            # MEM to 1st (state.WB["Wrt_data"])
            self.nextState.EX["nop"] = True
            self.nextState.ID["is_hazard"] = True
            return 3
        else:
            return 0
    
    def printState(self, state, cycle):
        printstate = ["-"*70 + "\n", "State after executing cycle: " + str(cycle) + "\n"]

        excluded_keys = ['is_hazard', 'instruction_count'] 
        
        
        for stage in ['IF', 'ID', 'EX', 'MEM', 'WB']:
            stage_state = getattr(state, stage)
            printstate.append("\n") 
            for key, val in stage_state.items():
                if key in excluded_keys:
                    continue  

                
                try:
                    if key == 'nop':
                        formatted_value = str(val)
                    elif key == 'Instr' and isinstance(val, str):
                        formatted_value = f"{int(val, 16):032b}" 
                    elif key in ['Read_data1', 'Read_data2', 'Imm', 'ALUresult', 'Store_data', 'Wrt_data']:
                        formatted_value = f"{val:032b}"  
                    elif key in ['Rs', 'Rt', 'Wrt_reg_addr']:
                        formatted_value = f"{val:05b}" 
                    elif key.endswith('_op') or key in ['is_I_type', 'rd_mem', 'wrt_mem', 'wrt_enable']:
                        formatted_value = str(val)  
                    elif isinstance(val, bool):  
                        formatted_value = str(val)
                    else:
                        formatted_value = str(val)  
                except ValueError as e:
                    formatted_value = str(val)  
                    print(f"Error converting {key}: {val} - {str(e)}")  

                
                printstate.append(f"{stage}.{key}: {formatted_value}\n")
        
        file_mode = "w" if cycle == 0 else "a"

        with open(self.opFilePath, file_mode) as wf:
            wf.writelines(printstate)



def WritetPerformanceMetrics(ioDir, cycles_ss, instr_count, CPI_ss, IPC_ss, cycles_fs, CPI_fs, IPC_fs):
    opFilePath = ioDir + os.sep + "PerformanceMetrics.txt"
    printstate_ss = ["Performance of Single Stage: \n"]
    printstate_ss.append("#Cycles -> " + str(cycles_ss) + "\n")
    printstate_ss.append("#Instructions -> " + str(instr_count) + "\n")
    printstate_ss.append("CPI -> " + str(CPI_ss) + "\n")
    printstate_ss.append("IPC -> " + str(IPC_ss) + "\n\n")

    printstate_fs = ["Performance of Five Stage: \n"]
    printstate_fs.append("#Cycles -> " + str(cycles_fs) + "\n")
    printstate_fs.append("#Instructions -> " + str(instr_count) + "\n")
    printstate_fs.append("CPI -> " + str(CPI_fs) + "\n")
    printstate_fs.append("IPC -> " + str(IPC_fs) + "\n\n")

    with open(opFilePath, 'w') as wf:
        wf.writelines(printstate_ss)
        wf.writelines(printstate_fs)

if __name__ == "__main__":

    #parse arguments for input file location
    parser = argparse.ArgumentParser(description='RV32I processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()

    ioDir = os.path.abspath(args.iodir)
    print("IO Directory:", ioDir)

    imem = InsMem("Imem", ioDir)
    dmem_ss = DataMem("SS", ioDir)
    dmem_fs = DataMem("FS", ioDir)

    ssCore = SingleStageCore(ioDir, imem, dmem_ss)
    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    # Single Stage

    while(True):
        if not ssCore.halted:
            ssCore.step()

        if ssCore.halted:
            break

    num_ss_cycles = ssCore.cycle
    instr_count = ssCore.state.IF['instruction_count'] - 1

    ss_IPC = instr_count / ssCore.cycle
    ss_CPI = 1/ss_IPC



    # Five Stage
  
    while(True):
        if not fsCore.halted:
           fsCore.step()

        if fsCore.halted: 
           break

    num_fs_cycles = fsCore.cycle
    

    fs_IPC = instr_count / fsCore.cycle
    fs_CPI = 1/fs_IPC



    #dump SS and FS data mem.
    dmem_ss.outputDataMem()
    dmem_fs.outputDataMem()

    # dump SS and FS Performance Metrics.
    # printPerformanceMetrics(ioDir, ss_CPI, ss_IPC, num_ss_cycles, fs_CPI, fs_IPC, num_fs_cycles)
    WritetPerformanceMetrics(ioDir, num_ss_cycles, instr_count, ss_CPI, ss_IPC, num_fs_cycles, fs_CPI, fs_IPC)