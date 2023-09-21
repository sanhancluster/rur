MODULE io_dice

    REAL(KIND=4), DIMENSION(:,:), ALLOCATABLE :: pos, vel
    REAL(KIND=4), DIMENSION(:), ALLOCATABLE :: mm
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: id
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: type
    !! HEAD VARIABLES
    TYPE htype
    	INTEGER*4, DIMENSION(6) :: npart
    	REAL*8, DIMENSION(6) :: mass
    	REAL*8 :: time
    	REAL*8 :: redshift
    	INTEGER*4 :: flag_sfr
    	INTEGER*4 :: flag_feedback
    	INTEGER*4, DIMENSION(6) :: nparttotal
    	INTEGER*4 :: flag_cooling
    	INTEGER*4 :: numfiles
    	REAL*8 :: boxsize
    	REAL*8 :: omega0
    	REAL*8 :: omegalambda
    	REAL*8 :: hubbleparam
    	INTEGER*4 :: flag_stellarage
    	INTEGER*4 :: flag_metals
    	INTEGER*4, DIMENSION(6)  :: totalhighword
    	INTEGER*4 :: flag_entropy_instead_u
    	INTEGER*4 :: flag_doubleprecision
    	INTEGER*4 :: flag_ic_info
    	REAL*4 :: lpt_scalingfactor
    	CHARACTER, DIMENSION(48) :: unused
    END TYPE htype

CONTAINS
    SUBROUTINE read_gadget(fname)

    IMPLICIT NONE
    CHARACTER(1000), INTENT(IN) :: fname

    !! LOCAL VARIABLES
    logical :: ok
    INTEGER :: dummy_int, blck_size, head_blck, np, ntype, i0, i1, i
    CHARACTER(LEN=4) :: blck_name


    TYPE(htype) :: header

    OPEN(unit=10, file=TRIM(fname), status='old', form='unformatted', action='read', access='stream')

    !! READ HEADER

    READ(10,POS=1+sizeof(dummy_int)) blck_name
    READ(10,POS=1+sizeof(dummy_int)+sizeof(blck_name)) dummy_int
    READ(10,POS=1+2*sizeof(dummy_int)+sizeof(blck_name)) dummy_int
    READ(10,POS=1+3*sizeof(dummy_int)+sizeof(blck_name)) blck_size
    head_blck = 1+sizeof(blck_name)+4*sizeof(dummy_int)

    READ(10,POS=head_blck) header%npart,header%mass,header%time,header%redshift, &
         header%flag_sfr,header%flag_feedback,header%nparttotal, &
         header%flag_cooling,header%numfiles,header%boxsize, &
         header%omega0,header%omegalambda,header%hubbleparam, &
         header%flag_stellarage,header%flag_metals,header%totalhighword, &
         header%flag_entropy_instead_u, header%flag_doubleprecision, &
         header%flag_ic_info, header%lpt_scalingfactor

    !PRINT *, header%npart

    !! READ PART
    np = sum(header%npart)

    CALL read_gadget_allocate(np)

    READ(10) pos
    READ(10) vel
    READ(10) id
    READ(10) mm
    CLOSE(10)

    i0 = 1
    i1 = 1
    DO i=1, SIZE(header%npart)
    	IF(header%npart(i) .EQ. 0) CYCLE
    	i1 = i0 + header%npart(i)-1
    	type(i0:i1) = i

    	i0 = i1 + 1
    ENDDO

    END SUBROUTINE read_gadget

!!!!!
    SUBROUTINE read_gadget_allocate(np)
    IMPLICIT NONE
    INTEGER(KIND=4) np

    IF(ALLOCATED(pos)) DEALLOCATE(pos)
    IF(ALLOCATED(vel)) DEALLOCATE(vel)
    IF(ALLOCATED(id)) DEALLOCATE(id)
    IF(ALLOCATED(mm)) DEALLOCATE(mm)
    IF(ALLOCATED(type)) DEALLOCATE(type)

    ALLOCATE(pos(1:3,1:np))
    ALLOCATE(vel(1:3,1:np))
    ALLOCATE(id(1:np))
    ALLOCATE(mm(1:np))
    ALLOCATE(type(1:np))
    END SUBROUTINE
!!!!!
    SUBROUTINE read_gadget_deallocate()
    IMPLICIT NONE
    IF(ALLOCATED(pos)) DEALLOCATE(pos)
    IF(ALLOCATED(vel)) DEALLOCATE(vel)
    IF(ALLOCATED(id)) DEALLOCATE(id)
    IF(ALLOCATED(mm)) DEALLOCATE(mm)
    IF(ALLOCATED(type)) DEALLOCATE(type)
    END SUBROUTINE


END MODULE io_dice