SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[APIData](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[api_endpoint] [varchar](50) NULL,
	[no_of_images_request] [int] NULL,
	[eclaims_record_id] [varchar](50) NULL,
	[sent_on_request] [datetime] NULL,
	[image_details_request] [nvarchar](max) NULL,
	[api_response] [nvarchar](max) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
ALTER TABLE [dbo].[APIData] ADD  CONSTRAINT [PK_APIRequestData] PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ONLINE = OFF, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO

ALTER TABLE [dbo].[APIData]
    ADD [is_response_correct] BIT            NULL,
        [user_feedback]       NVARCHAR (MAX) NULL;
GO